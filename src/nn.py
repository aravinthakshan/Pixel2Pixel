import torch
import torch.nn as nn
import torch.nn.init as init

class SimAttention(nn.Module):
    def __init__(self, channels):
        super(SimAttention, self).__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        attention = torch.mean(x_flat, dim=2, keepdim=True)
        attention = torch.sigmoid(attention).view(b, c, 1, 1)
        return x * attention


class EnhancedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EnhancedConvolution, self).__init__()
        padding = kernel_size // 2
        self.sim_attention = SimAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        mid_channels = max(out_channels // 4, 1)
        self.partial_conv = nn.Conv2d(out_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        attn_out = self.sim_attention(x)
        conv1_out = self.conv1(attn_out)
        partial_out = self.conv2(self.partial_conv(conv1_out))
        return conv1_out + partial_out


class ResidualREconvNet(nn.Module):
    def __init__(self):
        super(ResidualREconvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.reconv1 = EnhancedConvolution(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv_1x1_1 = nn.Conv2d(16, 32, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.reconv2 = EnhancedConvolution(32, 32, 5)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv_1x1_2 = nn.Conv2d(32, 32, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_1x1_3 = nn.Conv2d(32, 3, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        residual1 = out
        out = self.relu(self.bn2(self.reconv1(out))) + residual1
        out = self.relu(self.bn3(self.conv_1x1_1(out)))
        residual2 = out
        out = self.relu(self.bn4(self.reconv2(out))) + residual2
        out = self.relu(self.bn5(self.conv_1x1_2(out)))
        out = self.conv_1x1_3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = ResidualREconvNet()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {total_params:,}")
