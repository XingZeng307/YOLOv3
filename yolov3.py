from torch import nn
import torch
'''
config of model (transform the original config file into simplified version)
"D": Detection Block
"U": Upsample layer
"R": ResNet Block
"Y": Yolo Block
(32, 3, 1) --> (filters, kernel, stride)

original config file url: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
'''

config = [
    (32, 3, 1),
    (64, 3, 2),
    {'R': 1},
    (128, 3, 2),
    {'R': 2},
    (256, 3, 2),
    {'R': 8},  # Residual and save for Route
    (512, 3, 2),
    {'R': 8},  # Residual and save for Route
    (1024, 3, 2),
    {'R': 4},
    (512, 1, 1),
    "Y",  # Yolo Block and save for Route
    "D",  # Detection
    (256, 1, 1),
    "U",  # Upsample layer
    (256, 1, 1),
    "Y",  # Yolo Block and save for Route
    "D",  # Detection
    (128, 1, 1),
    "U",  # Upsample layer
    (128, 1, 1),
    "Y",  # Yolo Block and save for Route
    "D"  # Detection
]

'''
I divide YOLO network into different blocks including ConvBlock, ResBlock, YoloBlock and DetectionBlock. And
there are other things which are used such as Upsampling and  Route, order to concatenate feature maps from 
different layers
'''

# the whole architecture
class YOLO(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layers = self.create_model()

    def forward(self, x):
        cache = []  # save intermediate results for route layer
        results = []  # record the results in three scales
        for layer in self.layers:
            # print(layer)
            if isinstance(layer, DetectionBlock):
                # reshape into [batch, grid, grid, boxes, 5 + num_classes]
                results.append(
                    layer(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                    .permute(0, 3, 4, 1, 2)
                )
                continue
                
            x = layer(x)

            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                cache.append(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([x,  cache.pop()], dim=1)

        # return list.reverse(results)
        return results

    def create_model(self):
        layers = nn.ModuleList()
        in_channels = 3

        for ele in config:
            if isinstance(ele, tuple):
                out_channels = ele[0]
                padding_size = 1 if ele[1] == 3 else 0
                layers.append(ConvBlock(in_channels, out_channels, kernel_size=ele[1], padding=padding_size, stride=ele[2]))
                in_channels = out_channels

            # Residual Block
            elif isinstance(ele, dict):
                layers.append(ResBlock(in_channels, num_repeats=ele['R']))
                in_channels = in_channels

            # YoloBlock layer
            elif ele == 'Y':
                layers.append(YoloBlock(in_channels))
                in_channels = in_channels

            # upsample layer
            elif ele == 'U':
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels*3

            elif ele == "D":
                layers.append(DetectionBlock(in_channels, self.num_classes))
        return layers


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=False)
        self.use_bn = bn

    def forward(self, x):
        if self.use_bn:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=1, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        for i in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(in_channels, in_channels//2, kernel_size=1, padding=0),
                    ConvBlock(in_channels//2, in_channels, kernel_size=3, padding=1)
                )
            )

    def forward(self, x):
        if self.use_residual:
            for layer in self.layers:
                x = x + layer(x)
        else:
            for layer in self.layers:
                x = layer(x)
        return x


class YoloBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.yolo_layer = nn.Sequential(
            ConvBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            ConvBlock(in_channels*2, in_channels, kernel_size=1, padding=0),
            ConvBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            ConvBlock(in_channels*2, in_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.yolo_layer(x)


class DetectionBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.detection_layer = nn.Sequential(
                        ConvBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
                        ConvBlock(in_channels*2, 3*(4+1+num_classes), kernel_size=1, padding=0, bn=False)
                        # torch.Size([2, 255, 32, 32])
                    )

    def forward(self, x):
        return self.detection_layer(x)



if __name__ == "__main__":
    x = torch.randn((2, 3, 416, 416))
    model = YOLO(num_classes=20)
    out = model(x)
    # # out --> torch.Size([2, 13, 13,3, 85],[2, 26, 26, 3, 85],[2, 52, 52, 3, 85])

    # print(sum(p.numel() for p in model.parameters()))
    # print(model)
