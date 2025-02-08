from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vae import Decoder, Encoder

class VectorQuantizer2(nn.Module):
    def __init__(
        self, vocab_size, Cvae, using_znorm=False, beta=0.25, default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), quant_resi=0.5, share_quant_resi=4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.Cvae = Cvae
        self.using_znorm = using_znorm
        self.beta = beta
        self.v_patch_nums = v_patch_nums
        self.quant_resi = quant_resi
        self.share_quant_resi = share_quant_resi

        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        self.embedding.weight.data.uniform_(-1.0 / self.vocab_size, 1.0 / self.vocab_size)

    def f_to_idxBl_or_fhat(
        self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large

        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            if si != SN - 1:
                z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C)
            else:
                z_NC = f_rest.permute(0, 2, 3, 1).reshape(-1, C)

            # Ensure z_NC and embedding weight have the same data type
            z_NC = z_NC.float()  # Convert z_NC to Float
            embedding_weight_T = self.embedding.weight.data.T.float()  # Convert embedding weight to Float

            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(embedding_weight_T, dim=0), dim=1)
            else:
                d_no_grad = (
                    torch.sum(z_NC.square(), dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                )
                d_no_grad.addmm_(z_NC, embedding_weight_T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            if si != SN - 1:
                h_BChw = F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic'
                ).contiguous()
            else:
                h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()

            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

        return f_hat_or_idx_Bl

    def forward(self, f_BChw: torch.Tensor, ret_usages=False):
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl = self.f_to_idxBl_or_fhat(f_BChw, to_fhat=True)

        if ret_usages:
            usages = torch.bincount(f_hat_or_idx_Bl[-1].flatten(), minlength=self.vocab_size)
            return f_hat, usages, torch.tensor(0.0)  # Placeholder for vq_loss
        else:
            return f_hat, torch.tensor(0.0)  # Placeholder for vq_loss

class VQVAE(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
