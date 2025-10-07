import torch.nn as nn
import torch
import torch.nn.init as init

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=64, num_conv_layers=6, use_sigmoid = True):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.num_conv_layers = num_conv_layers
        self.use_sigmoid = use_sigmoid
        
        # First conv layer
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        
        # Middle conv layers (dynamically created)
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers - 2):  # -2 because we have conv1 and final conv
            self.conv_layers.append(nn.Conv2d(chan_embed, chan_embed, 3, padding=1))
        
        # Final conv layer (1x1)
        self.conv_final = nn.Conv2d(chan_embed, n_chan, 1)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        
        # Pass through all middle layers
        for conv_layer in self.conv_layers:
            x = self.act(conv_layer(x))
        
        x = self.conv_final(x)

        if self.use_sigmoid:
            return torch.sigmoid(x)
        
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