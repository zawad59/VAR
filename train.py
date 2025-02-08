import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam  # Import Adam optimizer

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume

def adjust_batch_size(args):
    """Reduce batch size if memory allocation fails"""
    args.batch_size = max(1, args.batch_size // 2)
    args.glb_batch_size = max(1, args.glb_batch_size // 2)
    print(f"[INFO] Adjusted batch sizes: global={args.glb_batch_size}, local={args.batch_size}")

def build_everything(args: arg_util.Args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, dataset_val = build_dataset(
            args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=max(1, round(args.batch_size * 1.5)),
            sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(),
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
                start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}')
    else:
        num_classes = 1000
        ld_val = ld_train = None
    
    from models import VAR, VQVAE, build_vae_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )

    # Create the base optimizer
    base_optimizer = Adam(var_wo_ddp.parameters(), lr=args.tlr)

    # Wrap the base optimizer with AmpOptimizer
    var_opt = AmpOptimizer(
        mixed_precision=args.fp16,  # Use args.fp16 for mixed precision
        optimizer=base_
