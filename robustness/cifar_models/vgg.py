'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False, injection=None, this_layer_input=None, this_layer_output=None, just_latent=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        if injection is None and this_layer_output is None and this_layer_input is None:
            out = self.features(x)
        else:
            # Sanity checks
            if this_layer_input and this_layer_input >=len(self.features):
                raise ValueError("Invalid layer to feed input to")
            if this_layer_output and this_layer_output >= len(self.features):
                raise ValueError("Invalid layer to extract output from")
            if this_layer_input and this_layer_output and this_layer_input >= this_layer_output:
                raise ValueError("Cannot extract output of layer before feeding in input")

            # Iterative pass through layers instead of direct call on sequential
            out = x
            for i, layer in enumerate(self.features):
                if this_layer_input:
                    if i < this_layer_input:
                        continue
                out = layer(out)
                if injection is not None and i == injection[0]:
                    with torch.no_grad():
                        out += injection[1].view_as(out)
                if this_layer_output is not None and i == this_layer_output:
                    wanted_latent = out
                    # Stop computation at latent space if only that is needed
                    if just_latent: return wanted_latent

        latent = out.view(out.size(0), -1)
        # Stop computation at latent space if only that is needed
        if just_latent: return latent

        out = self.classifier(latent)
        if this_layer_output != None:
            return out, wanted_latent
        if with_latent:
            return out, latent
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13', **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)


vgg11 = VGG11
vgg13 = VGG13
vgg16 = VGG16
vgg19 = VGG19
