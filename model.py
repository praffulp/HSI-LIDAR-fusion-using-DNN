import torch
import torch.nn as nn
from torch.nn import functional as F
from main import LIST_OF_PARAMETERS

FM = LIST_OF_PARAMETERS['fm']
NC = LIST_OF_PARAMETERS['nc']
Classes = LIST_OF_PARAMETERS['Classes']
input_channels = LIST_OF_PARAMETERS['ip_channel']

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = FM
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        op = self.project(res)
        return op


class ASPP_CNN(nn.Module):
    def __init__(self):
        super(ASPP_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = FM,
                out_channels = FM,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM*2, FM*4, 3, 1, 1),
            nn.BatchNorm2d(FM*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )

        self.aspp_layer1 = ASPP(NC, [6, 12, 18])
        self.aspp_layer2 = ASPP(input_channels, [6, 12, 18])

        self.out1 = nn.Sequential(
            # nn.Linear(FM*4, FM*2),
            nn.Linear(FM*4, Classes),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=FM,
                out_channels=FM,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(FM),  # BN can improve the accuracy by 4%-5%
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )

        self.out2 = nn.Sequential(
            # nn.Linear(FM*4, FM*2),
            nn.Linear(FM*4, Classes),
        )

        # self.out3 = nn.Linear(FM*4, Classes)  # fully connected layer, output 16 classes
        self.out3 = nn.Sequential(
            # nn.Linear(FM*4*2, Classes),
            # nn.ReLU(),
            nn.Linear(FM*4, Classes),
        )

    def forward(self, x1, x2):
        x1 = self.aspp_layer1(x1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = self.aspp_layer2(x2)
        x2 = self.conv4(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        x = x1 + x2

        out3 = self.out3(x)

        return out1, out2, out3
