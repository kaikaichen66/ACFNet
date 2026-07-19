import torch
import torch.nn as nn
from ultralytics.nn.modules import C3

__all__=['SCE', 'C2f_SCE', 'C3k2_SCE']

# 通道注意力模块 (Channel Attention Block)
class CAB(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块 (Spatial Attention Block)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# SCE (卷积多尺度增强注意力模块)
class SCE(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.in_channels = in_channels
        
        self.conv1X1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 多尺度深度可分离卷积
        self.dwconv3X3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU6(inplace=True))
        self.dwconv5X5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU6(inplace=True))
        self.dwconv7X7 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 7, 1, 3, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU6(inplace=True))
        self.dwconv9X9 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 9, 1, 4, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU6(inplace=True))

        # 通道与空间注意力组件
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多尺度融合
        x = self.dwconv3X3(x) + self.dwconv5X5(x) + self.dwconv7X7(x) + self.dwconv9X9(x)
        x = self.conv1X1(x)

        # 注意力计算
        c_att = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) + 
                             self.fc2(self.relu1(self.fc1(self.max_pool(x)))))
        
        s_att = self.sigmoid(self.conv_spatial(torch.cat([torch.mean(x, dim=1, keepdim=True), 
                                                         torch.max(x, dim=1, keepdim=True)[0]], dim=1)))

        return x * c_att + x * s_att

# 基础模块工具
def autopad(k, p=None, d=1):
    if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Bottleneck_SCE(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = SCE(c2)

    def forward(self, x):
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))

class C2f_SCE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_SCE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k_SCE(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SCE(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_SCE(C2f_SCE):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_SCE(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SCE(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
