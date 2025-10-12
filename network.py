import torch.nn as nn
import torch
import torch.nn.init as init

# class NetworkSyn(nn.Module):
#     def __init__(self, n_chan, chan_embed=64, num_conv_layers=6, use_sigmoid = True):
#         super(NetworkSyn, self).__init__()
#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.num_conv_layers = num_conv_layers
#         self.use_sigmoid = use_sigmoid
        
#         # First conv layer
#         self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        
#         # Middle conv layers (dynamically created)
#         self.conv_layers = nn.ModuleList()
#         for i in range(num_conv_layers - 2):  # -2 because we have conv1 and final conv
#             self.conv_layers.append(nn.Conv2d(chan_embed, chan_embed, 3, padding=1))
        
#         # Final conv layer (1x1)
#         self.conv_final = nn.Conv2d(chan_embed, n_chan, 1)
        
#         self._initialize_weights()

#     def forward(self, x):
#         x = self.act(self.conv1(x))
        
#         # Pass through all middle layers
#         for conv_layer in self.conv_layers:
#             x = self.act(conv_layer(x))
        
#         x = self.conv_final(x)

#         if self.use_sigmoid:
#             return torch.sigmoid(x)
        
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.orthogonal_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)

import torch
import torch.nn as nn
import torch.nn.init as init

class NetworkSyn(nn.Module):
    def __init__(self, n_chan, chan_embed=64, num_conv_layers=6, use_sigmoid=True):
        super(NetworkSyn, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder
        self.enc1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.enc2 = nn.Conv2d(chan_embed, chan_embed * 2, 3, stride=2, padding=1)  # downsample
        self.enc3 = nn.Conv2d(chan_embed * 2, chan_embed * 4, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(chan_embed * 4, chan_embed * 4, 3, padding=1)
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(chan_embed * 4, chan_embed * 2, 2, stride=2)
        self.dec1 = nn.Conv2d(chan_embed * 4, chan_embed * 2, 3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(chan_embed * 2, chan_embed, 2, stride=2)
        self.dec2 = nn.Conv2d(chan_embed * 2, chan_embed, 3, padding=1)
        
        # Final output
        self.out_conv = nn.Conv2d(chan_embed, n_chan, 1)
        
        self._initialize_weights()
    
    def forward(self, x):
        # --- Encoder ---
        e1 = self.act(self.enc1(x))        # [B, C, H, W]
        e2 = self.act(self.enc2(e1))       # [B, 2C, H/2, W/2]
        e3 = self.act(self.enc3(e2))       # [B, 4C, H/4, W/4]

        # --- Bottleneck ---
        b = self.act(self.bottleneck(e3))

        # --- Decoder ---
        d1 = self.act(self.up1(b))         # [B, 2C, H/2, W/2]
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.act(self.dec1(d1))

        d2 = self.act(self.up2(d1))        # [B, C, H, W]
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.act(self.dec2(d2))

        out = self.out_conv(d2)
        
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class NetworkReal(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(NetworkReal, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)