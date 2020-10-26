import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

def conv3x3(in_planes, out_planes, stride=1,dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation,bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=1, bias=False)

class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x 


class BasicBlock(nn.Module):
    expansion = 1
    tran_expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CSPBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2
    tran_expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,norm_layer=None):
        super(CSPBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.lrelu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        #print("x")
        
        out = self.conv1(x)

        #print("out")
        #print(type(out))

        out = self.bn1(out)
        out = self.lrelu(out)

        


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.lrelu(out)

        return out

class CSPBlock(nn.Module):

    def __init__(self, block, inplanes,blocks, stride=1, downsample=None, norm_layer=None, activation = None):
        super(CSPBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if activation is None:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = activation()
        
        self.inplanes = inplanes
        self.norm_layer = norm_layer
        
        self.crossstage =  nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=1, stride=1,bias=False)
        
        self.bn_crossstage = norm_layer(self.inplanes*2)
        
        ## first layer is different from others 
        if(self.inplanes <= 64):
            self.conv1 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1,bias=False)
            self.bn1 =  norm_layer(self.inplanes)
            self.layer_num = self.inplanes
        else:
            self.conv1 = nn.Conv2d(self.inplanes, self.inplanes*2, kernel_size=1, stride=1,bias=False)
            self.bn1 =  norm_layer(self.inplanes*2)
            self.layer_num = self.inplanes*2

         


        self.layers = self._make_layer(block, self.inplanes, blocks)
        
        self.trans = nn.Conv2d(self.inplanes*2, self.inplanes*2, kernel_size=1, stride=1, bias=False)
        

    def forward(self, x):
        cross = self.crossstage(x)
        cross = self.bn_crossstage(cross)
        
        cross = self.activation(cross)
        
        origin = self.conv1(x)
        
        origin = self.bn1(origin)
        
        origin = self.activation(origin)
        
        #print("origin")
        #print(type(origin))
        
        origin = self.layers(origin)
        #origin = self.trans(origin)

        #out = origin
        out = torch.cat((origin,cross), dim=1)

        return out
    
    def _make_layer(self, block, planes, blocks, stride=1):

        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.layer_num != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []

        if(self.inplanes <=64):
            layers.append(block(self.inplanes, planes, stride, downsample,norm_layer))
            self.inplanes = planes * block.expansion
        else:
            self.inplanes = planes * block.expansion
            layers.append(block(self.inplanes, planes, stride, downsample,norm_layer))
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,norm_layer=norm_layer))

        return nn.Sequential(*layers)

class CSPResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 norm_layer=None):
        super(CSPResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.lrelu = nn.LeakyReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)




        self.layer1 = CSPBlock(block, 64, layers[0], activation = nn.LeakyReLU) ## 256 out

        self.part_tran1 = self._make_tran(64,block.tran_expansion)

        self.layer2 = CSPBlock(block, 128, layers[1]-1, activation=Linear)
        
        self.part_tran2 = self._make_tran(128,block.tran_expansion)
        
        
        self.layer3 = CSPBlock(block, 256, layers[2]-1, activation = Linear)
        
        self.part_tran3 = self._make_tran(256,block.tran_expansion)

        self.layer4 = CSPBlock(block, 512, layers[3]-1, activation = nn.LeakyReLU)
        
        self.conv2 = nn.Conv2d(512*block.tran_expansion,512*2, kernel_size=1,stride=1,bias=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        #self.conv3 = nn.Conv2d(512*2,num_classes, kernel_size=1,stride=1)
        
        
        self.fn = nn.Linear(512*2,num_classes)
        
        for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
           elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
           for m in self.modules():
               if isinstance(m, CSPBottleneck):
                   nn.init.constant_(m.bn3.weight, 0)
               elif isinstance(m, BasicBlock):
                   nn.init.constant_(m.bn2.weight, 0)
                   
    def _make_tran(self, base,tran_expansion):
        return nn.Sequential(
            conv1x1(base*tran_expansion,base*2),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(),
            conv3x3(base*2, base*2, stride=2),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU()
        )
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.part_tran1(x)
        
        x = self.layer2(x)
        x = self.part_tran2(x)
        
        x = self.layer3(x)
        x = self.part_tran3(x)
        

        x = self.layer4(x)
        
        x = self.conv2(x)
        
        x = self.avgpool(x)
        
        
        x = x.view(-1,512*2)
        

        x = self.fn(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _cspresnet(arch, block, layers, pretrained,model_path, **kwargs):
    model = CSPResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model

def csp_resnet50(pretrained=False,model_path = "checkpoint.pt",**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet50', CSPBottleneck, [3, 4, 6, 3], pretrained,model_path = model_path,
                   **kwargs)


def csp_resnet101(pretrained=False,model_path = "checkpoint.pt", **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet101', CSPBottleneck, [3, 4, 23, 3], pretrained,model_path = model_path,
                   **kwargs)


def csp_resnet152(pretrained=False,model_path = "checkpoint.pt", **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cspresnet('cspresnet152', CSPBottleneck, [3, 8, 36, 3], pretrained, model_path = model_path,
                   **kwargs)



if __name__ == "__main__":
    net = csp_resnet152(pretrained=False,num_classes = 10)
    y = net(torch.randn(1, 3, 112, 112))
    print(y.size())   

