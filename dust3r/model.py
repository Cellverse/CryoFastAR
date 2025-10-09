# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed
from models.croco import CroCoNet  # noqa
from dust3r.spatial_memory import SpatialMemory
from functools import partial
from croco.models.blocks import Block
inf = float('inf')

class AsymmetricCroCo3DStereo (CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self,
                output_mode='pts3d',
                dec_type ='croco',
                head_type='linear',
                depth_mode=('exp', -inf, inf),
                conf_mode=('exp', 1, inf),
                freeze='none',
                landscape_only=True,
                patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                has_neck=False,
                has_sph=False,
                num_views=2,
                **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        croco_kwargs['dec_type'] = dec_type
        croco_kwargs['dec_num_views'] = num_views
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_type = dec_type
        self.num_views = num_views
        self.use_feat = False
        self.mem_pos_enc = False
        self.num_tokens = (128 // 16) ** 2 #  FIXME: make it adaptable


        if self.dec_type == 'croco':
            self.dec_blocks2 = deepcopy(self.dec_blocks)
        elif self.dec_type == 'mem':
            self.dec_blocks2 = deepcopy(self.dec_blocks)
            self.set_memory_encoder(enc_embed_dim=768 if self.use_feat else 1024, memory_dropout=0.15)
            self.set_attn_head()

        self.segment_embeddings = nn.Embedding(128, self.dec_embed_dim) #  FIXME: make it adaptable, 128 is the number of views

        self.initialize_weights()
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, has_neck=has_neck, num_views=num_views,has_sph=has_sph, **croco_kwargs)
        self.set_freeze(freeze)


    def initialize_weights(self):
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def set_memory_encoder(self, enc_depth=6, enc_embed_dim=1024, out_dim=1024, enc_num_heads=16,
                            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            memory_dropout=0.15):

        self.value_encoder = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True,
                    norm_layer=norm_layer, rope=self.rope if self.mem_pos_enc else None)
            for _ in range(enc_depth)])

        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)

        # Normalization layers
        self.norm_q = nn.LayerNorm(1024)
        self.norm_k = nn.LayerNorm(1024)
        self.norm_v = nn.LayerNorm(1024)
        self.mem_dropout = nn.Dropout(memory_dropout)

    def set_attn_head(self, enc_embed_dim=1024+768, out_dim=1024):
        self.attn_head_1 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

        self.attn_head_2 = nn.Sequential(
            nn.Linear(enc_embed_dim, enc_embed_dim),
            nn.GELU(),
            nn.Linear(enc_embed_dim, out_dim)
        )

    def encode_feat_key(self, feat1, feat2, num=1):
        feat = torch.cat((feat1, feat2), dim=-1)
        feat_k = getattr(self, f'attn_head_{num}')(feat)

        return feat_k

    def encode_value(self, x, pos):
        for block in self.value_encoder:
            x = block(x, pos)
        x = self.value_norm(x)
        x = self.value_out(x)
        return x

    def encode_cur_value(self, res1, dec1, pos1, shape1):
        if self.use_feat:
            cur_v = self.encode_value(dec1[-1], pos1)

        else:
            out, pos_v = self.pos_patch_embed(res1['pts3d'].permute(0, 3, 1, 2), true_shape=shape1)
            cur_v = self.encode_value(out, pos_v)

        return cur_v

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, 1, img_size, patch_size, enc_embed_dim)
        self.pos_patch_embed = get_patch_embed(self.patch_embed_cls, 3, img_size, patch_size, enc_embed_dim)


    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, has_neck, has_sph,
                            num_views, **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        self.downstream_heads = nn.ModuleList([
            head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), has_neck=has_neck, has_sph=has_sph) 
            for _ in range(2)
        ])
        # 使用 transpose_to_landscape 对每个 downstream_head 进行处理
        self.heads = [
            transpose_to_landscape(head, activate=landscape_only)
            for head in self.downstream_heads
        ]

        # re-initialize the last layer of heads to be zero
        for head in self.downstream_heads:
            nn.init.zeros_(head.proj.weight)
            nn.init.zeros_(head.proj.bias)
            nn.init.zeros_(head.proj_fvalue.weight)
            nn.init.zeros_(head.proj_fvalue.bias)



    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        # assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        if self.enc_type == 'croco':
            for blk in self.enc_blocks:
                x = blk(x, pos)
            x = self.enc_norm(x)

        return x, pos, true_shape

    def _encode_image_pairs(self, imgs, true_shapes):

        out, pos, _ = self._encode_image(torch.cat(imgs, dim=0),
                                        torch.cat(true_shapes, dim=0))
        outs = out.chunk(self.num_views, dim=0)
        positions = pos.chunk(self.num_views, dim=0)

        return outs, positions

    def _encode_symmetrized(self, views, is_symmetrized=False):
        imgs = [view['img'] for view in views] # (N, B, H, W, C)
        # img1 = view1['img']
        # img2 = view2['img']
        B = imgs[0].shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shapes = [view.get('true_shape', torch.tensor(view['img'].shape[-2:])[None].repeat(B, 1)) for view in views]
        # shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        # shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized:
            # computing 1/num_views of forward pass!'
            imgs = [img[::self.num_views] for img in imgs]
            shapes = [shape[::self.num_views] for shape in shapes]
            feats, positions = self._encode_image_pairs(imgs, shapes)
            feats = interleave(feats)
            positions = interleave(positions)
        else:
            feats, positions = self._encode_image_pairs(imgs, shapes)

        return shapes, feats, positions

    def encode_single_view(self, view):

        img = view['img'] # (B, H, W, C)
        B = img.shape[0]

        shape = view.get('true_shape', torch.tensor(view['img'].shape[-2:])[None].repeat(B, 1))

        return self._encode_image(img, shape)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _decoder_croco(self, feats, positions):

        features = []
        # if self.num_views > training_view:
        #     # interpolate self.segment_embeddings into more views
        #     segment_embeddings = self.segment_embeddings.weight[:training_view]
        #     # linear interpolation

        for i in range(self.num_views):
            f = self.decoder_embed(feats[i])
            B, P, _ = f.shape
            # if i >= training_view and i % training_view != training_view-1:
            #     i = i % training_view + 1
            # elif i >= training_view and i % training_view == training_view-1:
            #     i = training_view % training_view
            
            if i >= 128:
                i = i % 128
            idx = i * torch.ones(B, P,dtype=torch.long, device=f.device)
            segment_embedding = self.segment_embeddings(idx)

            f = f + segment_embedding

            features.append(f)

        final_output = []
        final_output.append(features)

        position_other_views = torch.cat(positions[1:], dim=1) # (B, (N-1)HW/P^2, 2)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            temp_features = []
            f1 = final_output[-1][0] # (B, L, D)
            f2 = torch.cat(final_output[-1][1:], dim=1) # (B, (N-1)L, D)
            # the first pass: data interaction with the first view, update view1's feature
            f1, _ = blk1(f1, f2, positions[0], position_other_views)
            temp_features.append(f1)
            # the second pass: data interaction with the other views, update other views' feature
            temps = []
            for i in range(1, self.num_views):
                fi = final_output[-1][i]
                posi = positions[i]
                # print(fi.shape, posi.shape)
                # img2 side
                fi = blk2.self_attn_block(fi, posi)
                temps.append(fi)

            f_other_views = torch.cat(temps, dim=1)
            f_other_views = blk2.cross_attn_block(f_other_views, f1, position_other_views, positions[0])
            # store the result
            temp_features += list(f_other_views.chunk(self.num_views - 1, dim=1))
            final_output.append(temp_features)


        # normalize last output
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _decoder_bert(self, f_list, pos_list):
        final_output = [f_list]

        input_embedding = []
        for i, f in enumerate(f_list):
            B, P, C = f.shape
            f = self.decoder_embed(f)
            idx = i * torch.ones(B, P,dtype=torch.long, device=f.device)

            segment_embedding = self.segment_embeddings(idx)

            embedding = f + segment_embedding
            input_embedding.append(embedding)

        final_output.append(input_embedding)

        f = self.norm0(torch.cat(input_embedding, dim=1))
        pos = torch.cat(pos_list, dim=1)

        for blk in self.dec_blocks:

            f = blk(f, pos)

            # split them back into f1, f2, ..., fn
            res = torch.chunk(f, self.num_views, dim=1)
            final_output.append(list(res))

        del final_output[1]  # duplicate with final_output[0]
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))

        head = self.heads[head_num]

        return head(decout, img_shape)

    def downstream_head(self, dec, true_shape, num=1):

        with torch.amp.autocast('cuda', enabled=False):
            res = self._downstream_head(num, [tok.float() for tok in dec], true_shape)

        return res

    def forward(self, views, is_symmetrized=False, encoder_only=False):
        # encode the two images --> B,S,D
        results = []
        if self.dec_type in ['croco', 'bert']:
            shapes, feats, positions = self._encode_symmetrized(views, is_symmetrized)
        
        if encoder_only:
            return feats
        # combine all ref images into object-centric representation
        if self.dec_type == 'croco':
            decs = self._decoder_croco(feats, positions)
        elif self.dec_type == 'bert':
            decs = self._decoder_bert(feats, positions)
        elif self.dec_type == 'mem':
            if self.training:
                sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v, mem_dropout=self.mem_dropout, attn_thresh=0)
            else:
                sp_mem = SpatialMemory(self.norm_q, self.norm_k, self.norm_v)

            feat1, feat2, pos1, pos2, shape1, shape2 = None, None, None, None, None, None
            feat_k1, feat_k2 = None, None

            for i in range(self.num_views - 1):
                reference_view = views[i]
                target_view = views[i + 1]

                # encode the two images --> B,S,D
                if i == 0:
                    feat1, pos1, shape1 = self.encode_single_view(reference_view)
                    feat2, pos2, shape2 = self.encode_single_view(target_view)
                    feat_fuse = feat1

                else:
                    feat1, pos1, shape1 = feat2, pos2, shape2
                    feat2, pos2, shape2 = self.encode_single_view(target_view)
                    feat_fuse = sp_mem.memory_read(feat_k2, res=True)

                dec1, dec2 = self._decoder(feat_fuse, pos1, feat2, pos2)

                #### Encode feat key
                feat_k1 = self.encode_feat_key(feat1, dec1[-1], 1)
                feat_k2 = self.encode_feat_key(feat2, dec2[-1], 2)

                with torch.amp.autocast('cuda', enabled=False):
                    res1 = self.downstream_head(dec1, shape1, 0)
                    res2 = self.downstream_head(dec2, shape2, 1)

                cur_v = self.encode_cur_value(res1, dec1, pos1, shape1)

                if self.training:
                    sp_mem.add_mem(feat_k1, cur_v + feat_k1)
                else:
                    sp_mem.add_mem_check(feat_k1, cur_v + feat_k1)

                results.append(res1)
            results.append(res2) # add the last one
        else:
            raise ValueError("Unsupported decoder type.")

        if self.dec_type in ['croco', 'bert']:
            decs = list(decs)
            with torch.amp.autocast('cuda', enabled=False):
                results = []
                for i in range(self.num_views):
                    if i == 0:
                        res = self.downstream_head(decs[i], shapes[0], 0)
                    else:
                        res = self.downstream_head(decs[i], shapes[1], 1)
                    results.append(res)

        return results, feats
