import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_2d(
    inp,
    oup,
    kernel_size=3,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    norm=True,
    act=True,
):
    conv = nn.Sequential()
    conv.add_module(
        "conv",
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups),
    )
    if norm:
        conv.add_module("BatchNorm2d", nn.BatchNorm2d(oup))
    if act:
        conv.add_module("Activation", nn.SiLU())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module(
                "exp_1x1", conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0)
            )
        self.block.add_module(
            "conv_3x3",
            conv_2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
            ),
        )
        self.block.add_module(
            "red_1x1",
            conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False),
        )
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MatMulModule(nn.Module):
    def __init__(self):
        super(MatMulModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))
        # Initialize any parameters or submodules if needed

    def forward(self, matrix_a, matrix_b):
        # Perform matrix multiplication
        return torch.matmul(matrix_a, matrix_b)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=4, dim_head=8, attn_dropout=0):
        super().__init__()
        self.mm = MatMulModule()
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.scale = dim_head**-0.5

    def forward(self, x):
        b_sz, S_len, in_channels = x.shape
        # self-attention
        # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
        qkv = self.qkv_proj(x).reshape(b_sz, S_len, 3, self.num_heads, -1)
        # [N, S, 3, h, c] --> [N, h, 3, S, C]
        qkv = qkv.transpose(1, 3).contiguous()
        # [N, h, 3, S, C] --> [N, h, S, C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q * self.scale
        # [N h, T, c] --> [N, h, c, T]
        k = k.transpose(-1, -2)
        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = self.mm(q, k)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = self.mm(attn, v)
        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self, embed_dim, ffn_latent_dim, heads=8, dim_head=8, dropout=0, attn_dropout=0
    ):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            Attention(embed_dim, heads, dim_head, attn_dropout),
            nn.Dropout(dropout),
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Module):
    def __init__(
        self, inp, attn_dim, ffn_multiplier, heads, dim_head, attn_blocks, patch_size
    ):
        super(MobileViTBlock, self).__init__()
        self.patch_h, self.patch_w = patch_size
        self.patch_area = int(self.patch_h * self.patch_w)

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(
            "conv_3x3", conv_2d(inp, inp, kernel_size=3, stride=1, padding=1)
        )
        self.local_rep.add_module(
            "conv_1x1",
            conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False),
        )

        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(
                f"TransformerEncoder_{i}",
                TransformerEncoder(attn_dim, ffn_dim, heads, dim_head),
            )
        self.global_rep.add_module(
            "LayerNorm", nn.LayerNorm(attn_dim, eps=1e-5, elementwise_affine=True)
        )

        self.conv_proj = conv_2d(attn_dim, inp, kernel_size=1, stride=1)
        self.fusion = conv_2d(2 * inp, inp, kernel_size=3, stride=1)

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(
                feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, self.patch_area
        )
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * self.patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return feature_map

    def forward(self, x):
        res = x.clone()
        x = self.local_rep(x)
        x, info_dict = self.unfolding(x)
        x = self.global_rep(x)
        x = self.folding(x, info_dict)
        x = self.conv_proj(x)
        x = self.fusion(torch.cat((res, x), dim=1))
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, mode, num_classes, patch_size=(2, 2)):
        super().__init__()
        # check image size
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0
        assert mode in ["xx_small", "x_small", "small"]

        # model size
        if mode == "xx_small":
            mv2_exp_mult = 2
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 16, 24, 48, 64, 80]
            attn_dim = [64, 80, 96]
        elif mode == "x_small":
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 32, 48, 64, 80, 96]
            attn_dim = [96, 120, 144]
        elif mode == "small":
            mv2_exp_mult = 4
            ffn_multiplier = 2
            last_layer_exp_factor = 4
            channels = [16, 32, 64, 96, 128, 160]
            attn_dim = [144, 192, 240]
        else:
            raise NotImplementedError

        self.conv_0 = conv_2d(3, channels[0], kernel_size=3, stride=2)

        self.layer_1 = nn.Sequential(
            InvertedResidual(
                channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult
            )
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(
                channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult
            ),
            InvertedResidual(
                channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult
            ),
            InvertedResidual(
                channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult
            ),
        )
        self.layer_3 = nn.Sequential(
            InvertedResidual(
                channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult
            ),
            MobileViTBlock(
                channels[3],
                attn_dim[0],
                ffn_multiplier,
                heads=4,
                dim_head=8,
                attn_blocks=2,
                patch_size=patch_size,
            ),
        )
        self.layer_4 = nn.Sequential(
            InvertedResidual(
                channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult
            ),
            MobileViTBlock(
                channels[4],
                attn_dim[1],
                ffn_multiplier,
                heads=4,
                dim_head=8,
                attn_blocks=4,
                patch_size=patch_size,
            ),
        )
        self.layer_5 = nn.Sequential(
            InvertedResidual(
                channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult
            ),
            MobileViTBlock(
                channels[5],
                attn_dim[2],
                ffn_multiplier,
                heads=4,
                dim_head=8,
                attn_blocks=3,
                patch_size=patch_size,
            ),
        )
        self.conv_1x1_exp = conv_2d(
            channels[-1], channels[-1] * last_layer_exp_factor, kernel_size=1, stride=1
        )
        self.out = nn.Linear(
            channels[-1] * last_layer_exp_factor, num_classes, bias=True
        )

    def forward(self, x):
        x = self.conv_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)

        # FF head
        x = torch.mean(x, dim=[-2, -1])
        x = self.out(x)

        return x
