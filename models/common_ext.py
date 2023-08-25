from torch import nn
import torchvision

_resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
_resnet34 = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
_shufflenet_v2_x2_0 = torchvision.models.shufflenet_v2_x2_0(
    weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT)


class _ResnetToLayer2(nn.Sequential):
    def __init__(self, resnet):
        super(_ResnetToLayer2, self).__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2


class Resnet18ToLayer2(_ResnetToLayer2):
    def __init__(self):
        super(Resnet18ToLayer2, self).__init__(_resnet18)


class Resnet34ToLayer2(_ResnetToLayer2):
    def __init__(self):
        super(Resnet34ToLayer2, self).__init__(_resnet34)


class Resnet18Layer3(nn.Sequential):
    def __init__(self):
        super(Resnet18Layer3, self).__init__()
        self.layer3 = _resnet18.layer3


class Resnet18Layer4(nn.Sequential):
    def __init__(self):
        super(Resnet18Layer4, self).__init__()
        self.layer4 = _resnet18.layer4


class Resnet34Layer3(nn.Sequential):
    def __init__(self):
        super(Resnet34Layer3, self).__init__()
        self.layer3 = _resnet34.layer3


class Resnet34Layer4(nn.Sequential):
    def __init__(self):
        super(Resnet34Layer4, self).__init__()
        self.layer4 = _resnet34.layer4


class ShuffleNetV2X20P3(nn.Sequential):
    MYEXT_OUT_CH = 244

    def __init__(self):
        super(ShuffleNetV2X20P3, self).__init__()
        self.conv1 = _shufflenet_v2_x2_0.conv1
        self.maxpool = _shufflenet_v2_x2_0.maxpool
        self.stage2 = _shufflenet_v2_x2_0.stage2


class ShuffleNetV2X20P4(nn.Sequential):
    MYEXT_OUT_CH = 488

    def __init__(self):
        super(ShuffleNetV2X20P4, self).__init__(_shufflenet_v2_x2_0.stage3)


class ShuffleNetV2X20P5(nn.Sequential):
    MYEXT_OUT_CH = 976

    def __init__(self):
        super(ShuffleNetV2X20P5, self).__init__(_shufflenet_v2_x2_0.stage4)


if __name__ == '__main__':
    import torch

    # model = ShuffleNetP3()
    # x = torch.rand(2, 3, 640, 640)
    # for nm, md in model.named_children():
    #     si = x.shape
    #     x = md(x)
    #     so = x.shape
    #     print(nm, si, so)
