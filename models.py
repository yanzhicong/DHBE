import os
import sys
import math

import torch
from torch import nn
import functools

import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import weight_norm

from torchvision.models.resnet import resnet18, resnet34, resnet50


def get_model(args, bn_args=None):

	if args.dataset in ["svhn", "cifar10", "cifar100"]:
		if args.model.lower() == "cnn13":
			return CNN13(num_classes=args.num_classes)
		elif args.model.lower() == "resnet18":
			return Resnet(base_encoder=resnet18, num_classes=args.num_classes, bn_args=bn_args)
		elif args.model.lower() == 'resnet34':
			return Resnet(base_encoder=resnet34, num_classes=args.num_classes, bn_args=bn_args)
		elif args.model.lower() == 'resnet50':
			return Resnet(base_encoder=resnet50, num_classes=args.num_classes, bn_args=bn_args)
		
		elif args.model.lower() == "wrn161":
			return wrn_16_1(num_classes=args.num_classes)
		elif args.model.lower() == "wrn162":
			return wrn_16_2(num_classes=args.num_classes)
		elif args.model.lower() == "wrn401":
			return wrn_40_1(num_classes=args.num_classes)
		elif args.model.lower() == "wrn402":
			return wrn_40_2(num_classes=args.num_classes)
		else:
			raise ValueError("unknown model : {}".format(args.model))


	elif args.dataset in ["vggface2_subset", "mini-imagenet"]:
		if args.model.lower() == "resnet18":
			return Resnet_64(base_encoder=resnet18, num_classes=args.num_classes, bn_args=bn_args)
		elif args.model.lower() == 'resnet34':
			return Resnet_64(base_encoder=resnet34, num_classes=args.num_classes, bn_args=bn_args)
		elif args.model.lower() == 'resnet50':
			return Resnet_64(base_encoder=resnet50, num_classes=args.num_classes, bn_args=bn_args)
		else:
			raise ValueError("unknown model : {}".format(args.model))



class Resnet(nn.Module):
	"""
		Resnet for CIFAR10: input size of 32x32x3
	"""
	def __init__(self, base_encoder=resnet18, num_classes=10, bn_args=None):
		super(Resnet, self).__init__()
		if bn_args is not None:
			norm_layer = functools.partial(nn.BatchNorm2d, **bn_args)
			encoder = base_encoder(pretrained=False, num_classes=num_classes, norm_layer=norm_layer)
		else:
			encoder = base_encoder(pretrained=False, num_classes=num_classes)
		encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		encoder.maxpool = nn.Identity()
		self.f = encoder

	def forward(self, x):
		if len(x.size()) == 4:
			return self.f(x)
		elif len(x.size()) == 3:
			return self.f(x.unsqueeze(0))[0]
		else:
			raise ValueError(str(x.size()))


	def _get_feature_internal(self, x):
		x = self.f.conv1(x)
		x = self.f.bn1(x)
		x = self.f.relu(x)
		x = self.f.maxpool(x)

		x = self.f.layer1(x)
		x = self.f.layer2(x)
		x = self.f.layer3(x)
		x = self.f.layer4(x)

		x = self.f.avgpool(x)
		x = torch.flatten(x, 1)

		return x, self.f.fc(x)

	def get_feature(self, x):
		if len(x.size()) == 4:
			return self._get_feature_internal(x)
		elif len(x.size()) == 3:
			f, logits = self._get_feature_internal(x.unsqueeze(0))
			return f[0], logits[0]
		else:
			raise ValueError(str(x.size()))

	def _get_intermediate_internal(self, x):
		ret_list = []
		x = self.f.conv1(x)
		x = self.f.bn1(x)
		x = self.f.relu(x)
		x = self.f.maxpool(x)

		x = self.f.layer1(x)
		ret_list.append(x)
		x = self.f.layer2(x)
		ret_list.append(x)
		x = self.f.layer3(x)
		ret_list.append(x)
		x = self.f.layer4(x)
		ret_list.append(x)

		x = self.f.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.f.fc(x)
		ret_list.append(x)

		return ret_list

	@property
	def fc(self):
		return self.f.fc

	def get_intermediate(self, x):
		if len(x.size()) == 4:
			return self._get_intermediate_internal(x)
		elif len(x.size()) == 3:
			ret_list = self._get_intermediate_internal(x.unsqueeze(0))
			return [r[0] for r in ret_list]
		else:
			raise ValueError(str(x.size()))



class Resnet_64(Resnet):
	"""
		Resnet for input size of 64x64x3
	"""
	def __init__(self, base_encoder=resnet18, num_classes=10, bn_args=None):
		super(Resnet, self).__init__()
		if bn_args is not None:
			norm_layer = functools.partial(nn.BatchNorm2d, **bn_args)
			encoder = base_encoder(pretrained=False, num_classes=num_classes, norm_layer=norm_layer)
		else:
			encoder = base_encoder(pretrained=False, num_classes=num_classes)
		encoder.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False)
		encoder.maxpool = nn.Identity()
		self.f = encoder
		self.update_bn_list()






class CNN13(nn.Module):
	   
	def __init__(self, num_classes=10):
		super(CNN13, self).__init__()
		self.encoder = nn.Sequential(
			weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(2, stride=2, padding=0),

			weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d(2, stride=2, padding=0),
			
			weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),
			weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),
			nn.AvgPool2d(6, stride=2, padding=0),
		)
		
		self.fc =  weight_norm(nn.Linear(128, num_classes))

	def forward(self, x):
		out = self.encoder(x)
		out = out.view(-1, 128)
		out = self.fc(out)
		return out

	def get_feature(self, x):
		out = self.encoder(x)
		out = out.view(-1, 128)
		return out, self.fc(out)






class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(WRNBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout( dropout_rate )
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WRNNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(WRNNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = WRNBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = WRNNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = WRNNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = WRNNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1,1))
        features = out.view(-1, self.nChannels)
        out = self.fc(features)

        if return_features:
            return out, features
        else:
            return out

def wrn_16_1(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_16_2(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)

def wrn_40_1(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_40_2(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)
