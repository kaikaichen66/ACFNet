import torch
import torch.nn.functional as F
import torch.nn as nn

__all__ = ['BSU', 'C3k2_BSC']

# --- 基础工具 ---
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# --- 核心组件 SSU (原 SRU) ---
class SSU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        gn_groups = group_num if oup_channels % group_num == 0 else 1
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=gn_groups)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.gate_treshold = nn.Parameter(torch.tensor(gate_treshold))
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / (self.gn.weight.sum() + 1e-6)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)
        x_1, x_2 = w1 * x, w2 * x
        c_half = x_1.size(1) // 2
        x_11, x_12 = x_1[:, :c_half], x_1[:, c_half:]
        x_21, x_22 = x_2[:, :c_half], x_2[:, c_half:]
        return torch.cat([x_11 * self.alpha + x_22 * (1 - self.alpha), 
                          x_12 * (1 - self.alpha) + x_21 * self.alpha], dim=1)

# --- 核心组件 CBE (原 CRU) ---
class CBE(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 0.5, squeeze_radio: int = 2):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel
        up_squeezed = max(1, self.up_channel // squeeze_radio)
        low_squeezed = max(1, self.low_channel // squeeze_radio)
        self.squeeze1 = nn.Conv2d(self.up_channel, up_squeezed, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, low_squeezed, kernel_size=1, bias=False)
        self.GWC = nn.Conv2d(up_squeezed, op_channel, kernel_size=3, padding=1, groups=2)
        self.PWC1 = nn.Conv2d(up_squeezed, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(low_squeezed, op_channel - low_squeezed, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.advmax = nn.AdaptiveMaxPool2d(1)
        self.final_conv = nn.Conv2d(op_channel * 2, op_channel, kernel_size=1, bias=False)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        pool_out = self.advavg(out) + self.advmax(out) 
        out = F.softmax(pool_out, dim=1) * out
        return self.final_conv(out)

# --- 整合模块 BSU ---
class BSU(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 4, gate_treshold: float = 0.5, alpha: float = 0.5):
        super().__init__()
        self.SSU = SSU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CBE = CBE(op_channel, alpha=alpha)

    def forward(self, x):
        return self.CBE(self.SSU(x))

# --- YOLO 集成模块 ---
class Bottleneck_BSC(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = BSU(c_)
        self.cv3 = Conv(c_, c2, k[1], 1, g=g) # 补全通道还原，确保 shortcut 正常工作
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k_BSC(nn.Module): # 修正：重写 C3k，避免继承带来的命名混乱
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 修正：调用 Bottleneck_BSC 而不是已不存在的 Bottleneck_ScConv
        self.m = nn.Sequential(*(Bottleneck_BSC(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2_BSC(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 核心引用：确保逻辑闭环
        self.m = nn.ModuleList(
            C3k_BSC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_BSC(self.c, self.c, shortcut, g) 
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    image = torch.rand(1, 64, 240, 240)
    model = C3k2_BSC(64, 64, n=1, c3k=True)
    out = model(image)
    print(f"--- BSU Module Ready ---")
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {out.size()}")