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
        mixed_precision=args.mixed_precision,  # Set this based on your args or config
        optimizer=base_optimizer,
        names=[name for name, param in var_wo_ddp.named_parameters() if param.requires_grad],
        paras=[param for param in var_wo_ddp.parameters() if param.requires_grad],
        grad_clip=args.grad_clip,  # Set this based on your args or config
        n_gradient_accumulation=args.n_gradient_accumulation,  # Set this based on your args or config
    )

    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var_wo_ddp,
        var_opt=var_opt, label_smooth=args.ls,  # Pass the AmpOptimizer here
    )
    
    return num_classes, ld_train, ld_val, start_ep, start_it, trainer

def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    adjust_batch_size(args)
    
    num_classes, ld_train, ld_val, start_ep, start_it, trainer = build_everything(args)
    
    start_time = time.time()
    for ep in range(start_ep, args.ep):
        torch.cuda.empty_cache()
        
        stats = train_one_ep(ep, ep == start_ep, start_it if ep == start_ep else 0, args, ld_train, trainer)
        
        print(f'[Epoch {ep}] Completed')
    
    print(f'  [*] [Training Finished]  Total cost: {((time.time() - start_time) / 60 / 60):.1f}h')
    
    gc.collect()
    torch.cuda.empty_cache()

def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, ld_train, trainer):
    from utils import misc
    me = misc.MetricLogger(delimiter='  ')
    
    for it, (inp, label) in enumerate(ld_train):
        inp, label = inp.to(args.device, non_blocking=True), label.to(args.device, non_blocking=True)
        
        with torch.amp.autocast("cuda"):
            # Ensure input tensors are in the correct data type
            inp = inp.float()  # Convert input to Float
            label = label.float()  # Convert label to Float
            
            trainer.train_step(
                it=it, g_it=ep * len(ld_train) + it, stepping=True,
                metric_lg=me, tb_lg=None, inp_B3HW=inp, label_B=label,
                prog_si=-1, prog_wp_it=max(1, args.pgwp * len(ld_train))
            )
    
    return {}

if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        torch.cuda.empty_cache()
