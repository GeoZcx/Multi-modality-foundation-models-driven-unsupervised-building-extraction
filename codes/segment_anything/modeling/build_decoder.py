import torch
from torch import nn, einsum

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class DeConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(DeConvBNReLU, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, bias=bias,
                               dilation=1, stride=2,
                               padding=0),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class CNN_Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(CNN_Stem, self).__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                       bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, norm_layer=norm_layer,
                       bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size=1, stride=1, norm_layer=norm_layer,
                       bias=bias)
        )

class CNN_Stem_2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bias=False):
        super(CNN_Stem_2, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)
        self.conv3 = ConvBNReLU(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
class Post_channel_mixer(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(Post_channel_mixer, self).__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, norm_layer=norm_layer,
                       bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, norm_layer=norm_layer,
                       bias=bias)
        )


class Post_channel_mixer(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(Post_channel_mixer, self).__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, norm_layer=norm_layer,
                       bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, norm_layer=norm_layer,
                       bias=bias)
        )

class Mdc_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=None):
        super().__init__()
        if dilation is None:
            dilation = [1, 2, 3]
        if in_channels == 256:
            mid_channels = 3*86
        elif in_channels == 128:
            mid_channels = 3*43
        elif in_channels == 64:
            mid_channels = 3*21
        elif in_channels == 32:
            mid_channels = 3*10
        elif in_channels == 16:
            mid_channels = 3*5

        self.conv1 = nn.Conv2d(in_channels=mid_channels//3, out_channels=mid_channels//3, kernel_size=3,
                               stride=1, padding=(dilation[0] * (3 - 1)) // 2, dilation=dilation[0])
        self.conv2 = nn.Conv2d(in_channels=mid_channels//3, out_channels=mid_channels//3, kernel_size=3,
                               stride=1, padding=(dilation[1] * (3 - 1)) // 2, dilation=dilation[1])
        self.conv3 = nn.Conv2d(in_channels=mid_channels//3, out_channels=mid_channels//3, kernel_size=3,
                               stride=1, padding=(dilation[2] * (3 - 1)) // 2, dilation=dilation[2])

        self.pre_channel_mixer = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.post_channel_mixer = Post_channel_mixer(in_channels=mid_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.pre_channel_mixer(x)

        # split x into 3 parts along the channel dimension
        # x1, x2, x3 = torch.split(x, x.shape[1] // 3, dim=1)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.post_channel_mixer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, *, C=256, num_classes=2):
        super().__init__()

        self.mdc_1 = Mdc_Block(in_channels=C, out_channels=C//2, dilation=[1, 2, 3])
        self.de_conv1 = DeConvBNReLU(in_channels=C//2, out_channels=C//2)

        self.conv_1 = ConvBNReLU(in_channels=C//2, out_channels=C//4, kernel_size=3, stride=1)
        self.conv_2 = ConvBNReLU(in_channels=C//4, out_channels=C//8, kernel_size=3, stride=1)
        self.conv_3 = ConvBNReLU(in_channels=C//8, out_channels=C//16, kernel_size=3, stride=1)

        self.mdc_2 = Mdc_Block(in_channels=C // 16, out_channels=num_classes, dilation=[1, 2, 3])

        self.mdc_3 = Mdc_Block(in_channels=C // 4, out_channels=C // 8, dilation=[1, 2, 3])
        self.de_conv2 = DeConvBNReLU(in_channels=C//4, out_channels=C//4)
        self.de_conv3 = DeConvBNReLU(in_channels=C//8, out_channels=num_classes)

        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):

        x = self.mdc_1(x)
        x = self.de_conv1(x) # 128
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.mdc_2(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Decoder(C=256, num_classes=2).cuda()
    summary(model.cuda(), input_size=[(1, 256, 64, 64)])
    input_1 = torch.randn(1,256,64,64).cuda()
    output = model(input_1)
    print(output.shape)