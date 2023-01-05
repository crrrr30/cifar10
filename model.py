# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.upsampling import Upsample


# class ImageLinearAttention(nn.Module):
    
#     def __init__(self, channels_in, channels_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8):
#         super().__init__()

#         self.channels_in = channels_in
#         channels_out = channels_in if channels_out is None else channels_out

#         self.key_dim = key_dim
#         self.value_dim = value_dim
#         self.heads = heads
        
#         conv_kwargs = {'padding': padding, 'stride': stride}
#         self.to_q = nn.Conv2d(channels_in, key_dim * heads, kernel_size, **conv_kwargs)
#         self.to_k = nn.Conv2d(channels_in, key_dim, kernel_size, **conv_kwargs)
#         self.to_v = nn.Conv2d(channels_in, value_dim, kernel_size, **conv_kwargs)

#         out_conv_kwargs = {'padding': padding}
#         self.to_out = nn.Conv2d(value_dim * heads, channels_out, kernel_size, **out_conv_kwargs)

#     def forward(self, x, context = None):
        
#         b, c, h, w, k_dim = *x.shape, self.key_dim

#         q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

#         q = q.reshape(b, self.heads, -1, h * w)
#         k = k.reshape(b, -1, h * w)
#         v = v.reshape(b, -1, h * w)

#         if context is not None:
#             context = context.reshape(b, c, 1, -1)
#             ck = self.to_k(context).reshape(b, k_dim, -1)
#             cv = self.to_v(context).reshape(b, k_dim, -1)
#             k = torch.cat((k, ck), dim = 2)
#             v = torch.cat((v, cv), dim = 2)

#         k = k.softmax(dim=2)
#         q = q.softmax(dim=2)

#         context = torch.einsum('bdn,ben->bde', k, v)
#         out = torch.einsum('bhdn,bde->bhen', q, context)
#         out = out.reshape(b, -1, h, w)
#         out = self.to_out(out)
        
#         return out


# def drop_connect(inputs, training, p=0.8):
    
#     if not training:
#         return inputs

#     batch_size = inputs.shape[0]
#     keep_prob = 1 - p

#     # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
#     random_tensor = keep_prob
#     random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
#     binary_tensor = torch.floor(random_tensor)

#     output = inputs / keep_prob * binary_tensor
#     return output


# class MBConvBlock(nn.Module):
    
#     def __init__(self, channels_in, channels_out, expand_ratio=6, use_se=True):
        
#         super().__init__()
#         self.use_se = use_se
        
#         _bn_mom = 0.01
#         _bn_eps = 0.001

#         self.channels_in = channels_in
#         self.channels_out = channels_out

#         # Expansion phase (Inverted Bottleneck)
#         self._expand_conv = nn.Conv2d(channels_in, channels_in * expand_ratio, 1, 1, 0, bias=False)
#         self._bn0 = nn.BatchNorm2d(num_features=channels_in * expand_ratio, momentum=_bn_mom, eps=_bn_eps)

#         # Depthwise convolution phase
#         self._depthwise_conv = nn.Conv2d(channels_in * expand_ratio, channels_in * expand_ratio, 3, 1, 1, groups=channels_in * expand_ratio)
#         self._bn1 = nn.BatchNorm2d(num_features=channels_in * expand_ratio, momentum=_bn_mom, eps=_bn_eps)

#         # Squeeze and excitation layer
#         if self.use_se:
#             num_squeezed_channels = max(1, channels_in // 4)
#             self._se_reduce = nn.Conv2d(channels_in * expand_ratio, num_squeezed_channels, 1, 1, 0)
#             self._se_expand = nn.Conv2d(num_squeezed_channels, channels_in * expand_ratio, 1, 1, 0)

#         # Pointwise convolution phase
#         final_oup = channels_out
#         self._project_conv = nn.Conv2d(in_channels=channels_in * expand_ratio, out_channels=final_oup, kernel_size=1, bias=False)
#         self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=_bn_mom, eps=_bn_eps)
        
#         self.relu = nn.ReLU(inplace=True)

#         if self.channels_in != self.channels_out:
#             self.add_conv = nn.Conv2d(self.channels_in, self.channels_out, 1, 1, 0, bias=False)

#     def forward(self, inputs):

#         x = inputs
        
#         # Expansion and Depthwise Convolution
#         x = self._expand_conv(inputs)
#         x = self._bn0(x)
#         x = self.relu(x)

#         x = self._depthwise_conv(x)
#         x = self._bn1(x)
#         x = self.relu(x)

#         # Squeeze and Excitation
#         if self.use_se:
#             x_squeezed = F.adaptive_avg_pool2d(x, 1)
#             x_squeezed = self._se_reduce(x_squeezed)
#             x_squeezed = self.relu(x_squeezed)
#             x_squeezed = self._se_expand(x_squeezed)
#             x = torch.sigmoid(x_squeezed) * x

#         # Pointwise Convolution
#         x = self._project_conv(x)
#         x = self._bn2(x)

#         # Connection
#         x = drop_connect(x, training=self.training)
#         if self.channels_in == self.channels_out:
#             x = x + inputs
#         else:
#             _ = self.add_conv(inputs)
#             x = x + _

#         return x


# class Model(nn.Module):

#     def __init__(self, attention = False):

#         super().__init__()

#         self.blocks = nn.ModuleList([
#             MBConvBlock(3, 16),
#             MBConvBlock(16, 32),
#             nn.MaxPool2d(2),
#             # ImageLinearAttention(32),
#             MBConvBlock(32, 64),
#             MBConvBlock(64, 128),
#             nn.MaxPool2d(2),
#             # ImageLinearAttention(128),
#             MBConvBlock(128, 256),
#             MBConvBlock(256, 512),
#             nn.MaxPool2d(2),
#             # ImageLinearAttention(512),
#             MBConvBlock(512, 512),
#             MBConvBlock(512, 512)
#         ])

#         self.linear = nn.Sequential(
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 10),
#             nn.Softmax(1)
#         )
    
#     def forward(self, x):

#         for block in self.blocks:
#             x = block(x)
        
#         x = x.mean((2, 3))
#         x = self.linear(x)

#         return x
        

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce


class SpatialGatingUnit(nn.Module):
    def __init__(self, n_tokens, dim):
        super().__init__()
        self.duplicate = nn.Linear(dim, dim * 2)
        self.ln = nn.LayerNorm(dim)
        self.dense = nn.Linear(n_tokens, n_tokens)

    def forward(self, x):
        x = self.duplicate(x)
        u, v = torch.chunk(x, 2, dim=-1)
        v = self.ln(v)
        v = self.dense(v.permute(0, 2, 1)).permute(0, 2, 1) + 1.      # Equiv. to bias init. = \vec 1
        return u * v


class gMLP(nn.Module):
    def __init__(self, n_tokens, d_model, d_ffn):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dense1 = nn.Linear(d_model, d_ffn)
        self.dense2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(n_tokens, d_ffn)

    def forward(self, x):
        shortcut = x
        x = self.ln(x)
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.sgu(x)
        x = self.dense2(x)
        return shortcut + x


class Model(nn.Module):
    def __init__(self, image_size, patch_size=4, n_layers=6, d_model=256, d_ffn=1024):
        super().__init__()
        self.patch_size = patch_size
        n_tokens = image_size * image_size // patch_size // patch_size

        self.patchify = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size),
            nn.Linear(3 * patch_size * patch_size, d_model)
        )
        self.backbone = nn.ModuleList([
            gMLP(n_tokens=n_tokens, d_model=d_model, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            Reduce("b n d -> b d", reduction="mean"),
            nn.Linear(d_model, 10)
        )
    
    def forward(self, x):
        x = self.patchify(x)
        for layer in self.backbone:
            x = layer(x)
        return self.classifier(x)

# model = Model(32)
# print(f"#params: {sum([w.numel() for w in model.parameters()]):,}")
# # print(model)
# x = torch.randn(4, 3, 32, 32)
# y = model(x)