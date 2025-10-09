import torch
from torch.nn import functional as F

class SpatialMemory():
    def __init__(self, norm_q, norm_k, norm_v, mem_dropout=None, 
                 long_mem_size=4000, work_mem_size=5, 
                 attn_thresh=5e-4, sim_thresh=0.95, 
                 save_attn=False):
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_v = norm_v
        self.mem_dropout = mem_dropout
        self.attn_thresh = attn_thresh
        self.long_mem_size = long_mem_size
        self.work_mem_size = work_mem_size
        self.top_k = long_mem_size
        self.save_attn = save_attn
        self.sim_thresh = sim_thresh
        self.init_mem()
    
    def init_mem(self):
        self.mem_k = None
        self.mem_v = None
        self.mem_c = None
        self.mem_count = None
        self.mem_attn = None
        self.mem_pts = None
        self.mem_imgs = None
        self.lm = 0
        self.wm = 0
        if self.save_attn:
            self.attn_vis = None

    def add_mem_k(self, feat):
        if self.mem_k is None:
            self.mem_k = feat
        else:
            self.mem_k = torch.cat((self.mem_k, feat), dim=1)

        return self.mem_k
    
    def add_mem_v(self, feat):
        if self.mem_v is None:
            self.mem_v = feat
        else:
            self.mem_v = torch.cat((self.mem_v, feat), dim=1)

        return self.mem_v

    def add_mem_c(self, feat):
        if self.mem_c is None:
            self.mem_c = feat
        else:
            self.mem_c = torch.cat((self.mem_c, feat), dim=1)

        return self.mem_c
    
    def add_mem_pts(self, pts_cur):
        if pts_cur is not None:
            if self.mem_pts is None:
                self.mem_pts = pts_cur
            else:
                self.mem_pts = torch.cat((self.mem_pts, pts_cur), dim=1)
    
    def add_mem_img(self, img_cur):
        if img_cur is not None:
            if self.mem_imgs is None:
                self.mem_imgs = img_cur
            else:
                self.mem_imgs = torch.cat((self.mem_imgs, img_cur), dim=1)

    def add_mem(self, feat_k, feat_v, pts_cur=None, img_cur=None):  
        if self.mem_count is None:
            self.mem_count = torch.zeros_like(feat_k[:, :, :1])
            self.mem_attn = torch.zeros_like(feat_k[:, :, :1])
        else:
            self.mem_count += 1
            self.mem_count = torch.cat((self.mem_count, torch.zeros_like(feat_k[:, :, :1])), dim=1)
            self.mem_attn = torch.cat((self.mem_attn, torch.zeros_like(feat_k[:, :, :1])), dim=1)
        
        self.add_mem_k(feat_k)
        self.add_mem_v(feat_v)
        self.add_mem_pts(pts_cur)
        self.add_mem_img(img_cur)
    
    def check_sim(self, feat_k, thresh=0.7):
        # Do correlation with working memory
        if self.mem_k is None or thresh==1.0:
            return False
        wmem_size = self.wm * 196

        # wm: BS, T, 196, C
        wm = self.mem_k[:, -wmem_size:].reshape(self.mem_k.shape[0], -1, 196, self.mem_k.shape[-1])

        feat_k_norm = F.normalize(feat_k, p=2, dim=-1)
        wm_norm = F.normalize(wm, p=2, dim=-1)

        corr = torch.einsum('bpc,btpc->btp', feat_k_norm, wm_norm)

        mean_corr = torch.mean(corr, dim=-1)

        if mean_corr.max() > thresh:
            print('Similarity detected:', mean_corr.max())
            return True
    
        return False

    def add_mem_check(self, feat_k, feat_v, pts_cur=None, img_cur=None):
        if self.check_sim(feat_k, thresh=self.sim_thresh):
            return
        
        self.add_mem(feat_k, feat_v, pts_cur, img_cur)
        self.wm += 1

        if self.wm > self.work_mem_size:
            self.wm -= 1
            if self.long_mem_size == 0:
                self.mem_k = self.mem_k[:, 196:]
                self.mem_v = self.mem_v[:, 196:]
                self.mem_count = self.mem_count[:, 196:]
                self.mem_attn = self.mem_attn[:, 196:]
                print('Memory pruned:', self.mem_k.shape)
            else:
                self.lm += 196 # TODO: Change this to the actual size of the memory bank
        
        if self.lm > self.long_mem_size:
            self.memory_prune()
            self.lm = self.top_k - self.wm * 196
    
    def memory_read(self, feat, res=True):
        '''
        Params:
            - feat: [bs, p, c]
            - mem_k: [bs, t, p, c]
            - mem_v: [bs, t, p, c]
            - mem_c: [bs, t, p, 1]
        '''
        
        affinity = torch.einsum('bpc,bxc->bpx', self.norm_q(feat), self.norm_k(self.mem_k.reshape(self.mem_k.shape[0], -1, self.mem_k.shape[-1])))
        affinity /= torch.sqrt(torch.tensor(feat.shape[-1]).float())
        
        if self.mem_c is not None:
            affinity = affinity * self.mem_c.view(self.mem_c.shape[0], 1, -1)  
        
        attn = torch.softmax(affinity, dim=-1)

        if self.save_attn:
            if self.attn_vis is None:
                self.attn_vis = attn.reshape(-1)
            else:
                self.attn_vis = torch.cat((self.attn_vis, attn.reshape(-1)), dim=0)
        if self.mem_dropout is not None:
            attn = self.mem_dropout(attn)
        
        if self.attn_thresh > 0:
            attn[attn<self.attn_thresh] = 0
            attn = attn / attn.sum(dim=-1, keepdim=True) 
        
        out = torch.einsum('bpx,bxc->bpc', attn, self.norm_v(self.mem_v.reshape(self.mem_v.shape[0], -1, self.mem_v.shape[-1])))
        
        if res:
            out = out + feat
        
        
        total_attn = torch.sum(attn, dim=-2)
        self.mem_attn += total_attn[..., None]
        
        return out
    
    def memory_prune(self):

        weights = self.mem_attn / self.mem_count
        weights[self.mem_count<self.work_mem_size+5] = 1e8

        num_mem_b = self.mem_k.shape[1]


        top_k_values, top_k_indices = torch.topk(weights, self.top_k, dim=1)
        top_k_indices_expanded = top_k_indices.expand(-1, -1, self.mem_k.size(-1))


        self.mem_k = torch.gather(self.mem_k, -2, top_k_indices_expanded)
        self.mem_v = torch.gather(self.mem_v, -2, top_k_indices_expanded)
        self.mem_attn = torch.gather(self.mem_attn, -2, top_k_indices)
        self.mem_count = torch.gather(self.mem_count, -2, top_k_indices)
 

        if self.mem_pts is not None:
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, 256, 3)
            self.mem_pts = torch.gather(self.mem_pts, 1, top_k_indices_expanded)
            self.mem_imgs = torch.gather(self.mem_imgs, 1, top_k_indices_expanded)

        num_mem_a = self.mem_k.shape[1]

        print('Memory pruned:', num_mem_b, '->', num_mem_a)