import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
import ipdb

# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, patch_loc, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = x
        
        hs, ap, ap_semantic = model.apEncoder(x_cond[:,0],patch_loc)
        rec = model.apDecoder(ap)
        mo = x_cond[:,1:]-x_cond[:,:-1]
       
        for i, j in zip(reversed(seq), reversed(seq_next)):
            ap_hs = hs.copy()
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs
            et = model.diffUNet(xt,mo,ap_hs,ap_semantic,t)
            et = et.unsqueeze(1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs = xt_next
    return xs, rec



def generalized_steps_womerge(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        # xs = torch.cat([crop(x, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0).to('cuda')
        x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
        xs = torch.randn_like(x_cond_patch[:,:1]).to('cuda')
        corners = (torch.tensor(corners).to('cuda'))/p_size
        patch_loc = torch.cat([corners.unsqueeze(1) for _ in range(n)], dim=1).reshape(-1,2)
        hs, ap, ap_semantic = model.apEncoder(x_cond_patch[:,0], patch_loc)
        rec = model.apDecoder(ap)
        rec_loss = (rec - x_cond_patch[:,0]).square()
        mo = x_cond_patch[:,1:]-x_cond_patch[:,:-1]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            ap_hs = hs.copy()
            t = (torch.ones(1) * i).to(x.device)
            next_t = (torch.ones(1) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs
            et = model.diffUNet(xt,mo,ap_hs,ap_semantic,t)
            et = et.unsqueeze(1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et    
            xs = xt_next
        rec_loss = rec_loss.mean([1,2,3]).reshape(-1,n).permute(1,0).max(1)[0]
    return xs,rec_loss


def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = x.to('cuda')
        # manual_batching_size = 64
        x_grid_mask = torch.zeros_like(x, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, :, hi:hi + p_size, wi:wi + p_size] += 1
        x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
        patch_loc = (torch.tensor(corners).to('cuda'))/p_size
        patch_loc = torch.cat([patch_loc.unsqueeze(1) for _ in range(n)], dim=1).reshape(-1,2)
        hs, ap, ap_semantic = model.apEncoder(x_cond_patch[:,0], patch_loc)
        rec = model.apDecoder(ap)
        rec_loss = (rec - x_cond_patch[:,0]).square()
        mo = x_cond_patch[:,1:]-x_cond_patch[:,:-1]
        # ipdb.set_trace()
        for i, j in zip(reversed(seq), reversed(seq_next)):
            ap_hs = hs.copy()
            t = (torch.ones(1) * i).to(x.device)
            next_t = (torch.ones(1) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs
            et_output = torch.zeros_like(xt, device=x.device)    
            xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            outputs = model.diffUNet(xt_patch,mo,ap_hs,ap_semantic,t)
            
            for idx, (hi, wi) in enumerate(corners):
                et_output[:, 0, :,hi:hi + p_size, wi:wi + p_size] += outputs.reshape(-1,n,3,p_size,p_size)[idx]
            # ipdb.set_trace()
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs = xt_next
            
        rec_loss = rec_loss.mean([1,2,3]).reshape(-1,n).permute(1,0).max(1)[0]
    return xs,rec_loss
