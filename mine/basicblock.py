from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''


def sequential(*args):
    """
    根据输入参数创建一个顺序模型。该函数支持将多个模块按顺序组合成一个nn.Sequential容器。

    参数:
    *args: 一个或多个torch.nn.Module对象，或者一个nn.Sequential对象。

    返回:
    如果只有一个参数且为OrderedDict类型，则直接返回该参数。
    如果有多个参数，将它们组合成一个nn.Sequential对象并返回。

    注意:
    如果输入为OrderedDict类型，将抛出NotImplementedError异常，因为有序字典输入目前不被支持。
    """
    # 当只有一个参数且为OrderedDict类型时，直接返回该参数，不进行后续处理。
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):  # 用于检查一个对象是否是特定类的实例，或者是特定类的子类的实例。
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.

    modules = []
    # 遍历所有参数，将它们按顺序添加到modules列表中。
    for module in args:
        # 如果参数是nn.Sequential类型，将其子模块逐个添加到modules列表中。
        if isinstance(module, nn.Sequential):
            for submodule in module.children(): # children() 只返回直接子模块，即那些作为构造函数参数传递给父模块的子模块
                modules.append(submodule)
        # 如果参数是其他类型的nn.Module，直接添加到modules列表中。
        elif isinstance(module, nn.Module):
            modules.append(module)

    # 最终将modules列表作为一个整体传入nn.Sequential，创建一个顺序模型并返回。
    return nn.Sequential(*modules)



'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''


# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    """
       构建一个卷积模块，根据配置模式(mode)执行不同的操作。

       参数:
       - in_channels: 输入通道数
       - out_channels: 输出通道数
       - kernel_size: 卷积核大小
       - stride: 步长
       - padding: 填充大小
       - bias: 是否使用偏置
       - mode: 模块操作的配置模式，每个字符代表一个操作

       返回:
       - 一个由指定操作组成的Sequential模块
       """
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class ConditionalBatchNorm2d(nn.Module):
    """
        条件批量归一化2d类，根据不同的类别条件进行批量归一化操作。

        参数:
        - num_features: 每个输入样本的特征数。
        - num_classes: 输入样本的类别数，用于条件归一化。
        """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


'''
FFT block
'''

class FFTBlock(nn.Module):
    """
       FFT处理块，继承自nn.Module。
       该块主要进行傅里叶变换相关的操作，包括卷积和复数除法等。

       参数:
       - channel (int): 卷积层的通道数，默认为64。
       """
    def __init__(self, channel=64):
        super(FFTBlock, self).__init__()
        self.conv_fc = nn.Sequential(
                nn.Conv2d(1, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Softplus()
        )

    def forward(self, x, u, d, sigma):
        """
           前向传播函数，执行FFT处理的主要逻辑。

           参数:
           - x (Tensor): 输入数据，形状为 [batch_size, channels, height, width]。
           - u (Tensor): 傅里叶变换后的系数，形状与 x 相同。
           - d (Tensor): 频率域的系数，形状与 x 相同。
           - sigma (Tensor): 噪声估计值，形状与 x 的通道数相同。

           返回:
           - x (Tensor): 处理后的输出数据，形状与输入 x 相同。
           """
        rho = self.conv_fc(sigma)
        x = torch.irfft(self.divcomplex(u + rho.unsqueeze(-1)*torch.rfft(x, 2, onesided=False), d + self.real2complex(rho)), 2, onesided=False)
        return x

    def divcomplex(self, x, y):
        """
            复数除法函数，用于在傅里叶变换中处理复数。

            参数:
            - x (Tensor): 复数数组x，形状为 [..., 2]，其中最后一个维度表示复数的实部和虚部。
            - y (Tensor): 复数数组y，形状与 x 相同。

            返回:
            - result (Tensor): 两个复数相除的结果，形状与输入相同。
            """
        a = x[..., 0]     # 分别获取x和y的实部和虚部
        b = x[..., 1]
        c = y[..., 0]
        d = y[..., 1]
        cd2 = c**2 + d**2   # 计算分母的模的平方
        return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)

    def real2complex(self, x):
        return torch.stack([x, torch.zeros(x.shape).type_as(x)], -1)




# -------------------------------------------------------
# Concat the output of a submodule to its input
# -------------------------------------------------------
class ConcatBlock(nn.Module):
    """
       一个包装块，用于将输入和其通过子模块处理后的结果在通道维度上进行拼接。

       Attributes:
           sub (nn.Module): 被包装的子模块。
       """
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        """
                返回模块的字符串表示，用于打印模块信息。

                返回:
                    str: 模块的字符串表示，末尾附加'concat'以表示拼接操作。
                """
        return self.sub.__repr__() + 'concat'


# -------------------------------------------------------
# Elementwise sum the output of a submodule to its input
# -------------------------------------------------------
class ShortcutBlock(nn.Module):
    """
        实现一个快捷连接块，继承自nn.Module。

        快捷连接块的作用是在神经网络中将输入进行变换后与原始输入相加，
        以实现残差学习或跳跃连接，有助于缓解深层网络的梯度消失问题。

        参数:
        - submodule: 子模块，是一个nn.Module对象，用于对输入进行变换。

        返回:
        - output: 输入与子模块变换后的输入相加的结果。
        """
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        # 返回模块的字符串表示，用于打印模块信息
        tmpstr = 'Identity + \n|'
        # 替换子模块的字符串表示中的换行符，以便正确格式化
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    """
    实现一个残差块，其中包含一个卷积层及其激活函数和归一化操作（如果有的话）。

    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - kernel_size: 卷积核大小
    - stride: 卷积步长
    - padding: 卷积padding
    - bias: 是否使用偏置
    - mode: 模式字符串，定义了卷积层的配置，包括激活函数和归一化操作的顺序。

    注意: 目前只支持输入输出通道数相同的情况。
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        # 确保输入和输出通道数相同，因为残差块通常是在相同维度上进行的
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        # 如果模式字符串以'R'或'L'开头，则将其转换为小写，以保持模式字符串的一致性
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        # 根据给定的模式和参数初始化卷积层
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        """
        前向传播函数，实现残差块的特征计算。

        参数:
        - x: 输入特征图

        返回:
        - 输入特征图与经过卷积层后的特征图相加的结果，体现了残差学习的思想。
        """
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    """
    通道注意力机制层（Channel Attention Layer）。

    该层通过自适应平均池化和全连接（FC）层模拟通道间的相互依赖关系，
    从而增强有效特征并抑制不相关特征。

    参数:
    - channel (int): 输入通道数。默认为64。
    - reduction (int): 通道缩减比例。默认为16，表示FC层中间输出的通道数为输入通道数的1/16。
    """
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()
        # 自适应平均池化，将HxW维度压缩到1，以便进行通道注意力计算
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 两个卷积层组成的序列模型，用于通道注意力的计算
        # 第一个卷积层减少通道数，第二个卷积层恢复通道数
        # ReLU激活函数增加非线性，Sigmoid激活函数生成通道权重
        self.conv_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
        前向传播过程，计算输入张量的通道注意力加权。
        参数:
        - x (Tensor): 输入张量。
        返回:
        - Tensor: 经过通道注意力加权后的输出张量。
        """
        # 对输入进行自适应平均池化，获取每个通道的特征图
        y = self.avg_pool(x)
        # 通过卷积序列模型计算通道注意力
        y = self.conv_fc(y)
        # 将原始输入与计算得到的注意力相乘，实现通道加权
        return x * y


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        """
        前向传播函数，实现RCAB的特征提取和残差连接。
        参数:
        - x: 输入张量。
        返回:
        经过RCAB模块处理后的输出张量，通过残差连接将输入与输出特征图相加。
        """
        # 通过残差单元和通道注意力层，最后与输入相加实现残差连接
        res = self.res(x)
        res = self.ca(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 reduction=16, nb=12):
        """
        初始化函数，构建残差通道注意力组（RCAGroup）的实例

        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - kernel_size: 卷积核大小
        - stride: 卷积步长
        - padding: 卷积padding
        - bias: 卷积层是否使用偏置
        - mode: 模式，如'CRC'表示卷积->激活->卷积
        - reduction: 注意力模块中通道压缩比例
        - nb: 每个RG中RCABlock的数量
        """
        super(RCAGroup, self).__init__()
        # 确保输入和输出通道数相同，以便残差连接
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        # 如果模式以'R'或'L'开头，调整为小写以保持一致性
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        # 构建RG序列，首先添加nb个RCABlock，然后添加一个卷积层用于最终的通道调整
        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction) for _ in
              range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        # 将RG序列封装为一个层，准备用于前向传播
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        """
        前向传播函数，执行一次RCAGroup的前向计算

        参数:
        - x: 输入张量

        返回:
        - 输出张量，经过RG处理后的结果
        """
        # 通过RG模块进行前向传播，得到残差张量
        res = self.rg(x)
        # 将输入x与残差张量res相加，实现残差连接
        return res + x


# -------------------------------------------------------
# 残差密集块
# 风格：5个卷积层
# -------------------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        """
        初始化残差密集块，包含5个卷积层。

        参数:
        nc (int): 输入和输出通道的数量。
        gc (int): 每个卷积层增长的通道数。
        kernel_size (int): 卷积核的大小。
        stride (int): 卷积的步长。
        padding (int): 在输入的四周添加的填充数量。
        bias (bool): 如果为True，在卷积中添加可学习的偏置项。
        mode (str): 卷积模式，'CR'表示卷积和激活。
        """
        super(ResidualDenseBlock_5C, self).__init__()

        # gc: 增长通道数
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1])

    def forward(self, x):
        """
        残差密集块的前向传播。

        参数:
        x (Tensor): 输入张量。

        返回:
        Tensor: 经过残差密集块处理后的输出张量。
        """
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 加权残差连接
        return x5.mul_(0.2) + x


# -------------------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# -------------------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        """
        初始化残差中的残差密集块（RRDB）。

        参数:
        - nc: 输入和输出通道的数量。
        - gc: 密集块中每一层增长的通道数。
        - kernel_size: 卷积核的大小。
        - stride: 卷积的步长。
        - padding: 在输入两侧添加的填充。
        - bias: 如果为True，在卷积中添加可学习的偏置。
        - mode: 卷积层配置，'C' 表示卷积，'R' 表示 PReLU 激活。
        """
        super(RRDB, self).__init__()

        # 初始化三个具有相同配置的残差密集块
        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        """
        RRDB 的前向传播。

        参数:
        - x: 输入张量。

        返回:
        - 经过 RRDB 后的输出张量，残差部分乘以 0.2 的缩放因子后与输入相加。
        """
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # 将残差乘以 0.2 来稳定梯度流并提高模型稳定性
        return out.mul_(0.2) + x


'''
# ======================
# Upsampler
# ======================
'''


# -------------------------------------------------------
# conv + subp + relu   实现图像的上采样操作，特别适用于超分辨率任务。

# mode: 上采样模式，'2'、'3'、'4'表示放大倍数，可附加'R'表示使用ReLU，'B'表示使用BatchNorm
# -------------------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    # 根据mode参数构建卷积层，注意输出通道数会乘以mode值的平方
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode)
    return up1


# -------------------------------------------------------
# nearest_upsample + conv + relu
# -------------------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode)
    return up1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1


'''
# ======================
# Downsampler
# ======================
'''


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1


# -------------------------------------------------------
# maxpooling + conv + relu
# -------------------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R'):
    """
    下采样模块，包含最大池化、卷积和ReLU激活。

    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - kernel_size: 卷积核大小
    - stride: 步长
    - padding: 填充大小
    - bias: 卷积层是否使用偏置
    - mode: 模式字符串，示例: '2', '2R', '2BR', '3', ..., '3BR'

    返回:
    - sequential: 一个包含池化和卷积操作的序列模块
    """
    # 确保mode的长度和首字符是合法的
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    # 根据mode的首字符确定池化层的kernel_size和stride
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    # 将mode中的首字符替换为'MC'，以适应后续的conv函数调用
    mode = mode.replace(mode[0], 'MC')
    # 构建池化层
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    # 构建池化后的尾部卷积层
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    # 返回由池化层和卷积层组成的序列模块
    return sequential(pool, pool_tail)


# -------------------------------------------------------
# averagepooling + conv + relu
# -------------------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


'''
# ======================
# NonLocalBlock2D: 
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# ======================
'''


# -------------------------------------------------------
# embedded_gaussian
# -------------------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool'):
        """
        初始化非局部2D块。

        参数:
        - nc: int, 通道数。
        - kernel_size: int, 卷积核大小。
        - stride: int, 卷积步长。
        - padding: int, 填充大小。
        - bias: bool, 是否使用偏置。
        - act_mode: str, 激活模式。
        - downsample: bool, 是否下采样。
        - downsample_mode: str, 下采样模式。
        """
        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        """
        前向传播。

        参数:
        - x: 输入张量，形状为(b, c, t, h, w)。

        返回:
        - z: 输出张量。
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z