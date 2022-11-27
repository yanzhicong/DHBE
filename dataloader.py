import os
import sys
import numpy as np
from torch.utils import data

from torchvision import datasets, transforms
import torch
from backdoor.base import SimpleDataset, ImageFolderDataset



def get_mean_std(args):
	if args.dataset.lower()=='mnist':
		return {"mean":[0.1307,], "std":[0.3081,]}

	elif args.dataset.lower()=='svhn':
		return {"mean":[0.43768454, 0.44376847, 0.4728039 ], "std":[0.19803019, 0.20101567, 0.19703582]}

	elif args.dataset.lower()=='cifar10':
		return {"mean":[0.4914, 0.4822, 0.4465], "std":[0.2023, 0.1994, 0.2010]}

	elif args.dataset.lower()=='cifar100':
		return {"mean":[0.5071, 0.4867, 0.4408], "std":[0.2675, 0.2565, 0.2761]}

	elif args.dataset.lower()=='vggface2_subset':
		return {"mean":[0.54452884, 0.43393408, 0.37677036,], "std":[0.28070204, 0.25066751, 0.24151797,]}

	elif args.dataset.lower()=='mini-imagenet':
		return {"mean":[0.47276672, 0.4485651, 0.40345553], "std":[0.26598931, 0.25850098, 0.27272259]}

	elif args.dataset.lower() == 'imagenet':
		return {"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]}
	else:
		raise ValueError()



def get_norm_trans(args):
	return transforms.Normalize(**get_mean_std(args))


def get_norm_trans_inv(args):
	def get_inv_mean_std(mean, std):
		mean = np.array(mean)
		std = np.array(std)
		mean_inv = - mean / std
		std_inv = 1.0 / std
		return {'mean' : mean_inv, 'std' : std_inv}	
	return transforms.Normalize(**get_inv_mean_std(**get_mean_std(args)))





def get_dataset(args):
	if args.dataset.lower()=='mnist':
		train_dataset = datasets.MNIST("/home/zhicong/data", train=True, download=True,
					   transform=transforms.Compose([
						   transforms.Resize((32, 32)),
						   transforms.ToTensor(),
						   get_norm_trans(args),
						]))
		test_dataset = datasets.MNIST("/home/zhicong/data", train=False, download=True,
					  transform=transforms.Compose([
						  transforms.Resize((32, 32)),
						  transforms.ToTensor(),
						  get_norm_trans(args),
						]))



	elif args.dataset.lower()=='svhn':
		train_dataset = datasets.SVHN("/home/zhicong/data", split="train", download=True,
						transform=transforms.Compose([
								transforms.RandomCrop(32, padding=4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								get_norm_trans(args),
							]))

		test_dataset = datasets.SVHN("/home/zhicong/data", split="test", download=True,
					   transform=transforms.Compose([
							transforms.ToTensor(),
							get_norm_trans(args),
						]))
		train_dataset = SimpleDataset(train_dataset.data.transpose([0, 2, 3, 1]), train_dataset.labels, train_dataset.transform)
		test_dataset = SimpleDataset(test_dataset.data.transpose([0, 2, 3, 1]), test_dataset.labels, test_dataset.transform)



	elif args.dataset.lower()=='cifar10':
		train_dataset = datasets.CIFAR10("/home/zhicong/data", train=True, download=True,
						transform=transforms.Compose([
								transforms.RandomCrop(32, padding=4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								get_norm_trans(args),
							]))

		test_dataset = datasets.CIFAR10("/home/zhicong/data", train=False, download=True,
					   transform=transforms.Compose([
							transforms.ToTensor(),
							get_norm_trans(args),
						]))



	elif args.dataset.lower()=='cifar100':
		train_dataset = datasets.CIFAR100("/home/zhicong/data", train=True, download=True,
					   transform=transforms.Compose([
							transforms.RandomCrop(32, padding=4),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							get_norm_trans(args),
						]))

		test_dataset = datasets.CIFAR100("/home/zhicong/data", train=False, download=True,
					   transform=transforms.Compose([
							transforms.ToTensor(),
							get_norm_trans(args),
						]))


	elif args.dataset.lower()=='vggface2_subset':
		train_dataset = ImageFolderDataset("/mnt/ext/zhicong/VGGface2/train_cls_subset", 
						preprocess_before = transforms.Compose([
							transforms.Resize(84),
							transforms.CenterCrop(64),
						]),
					   transform=transforms.Compose([
							transforms.RandomCrop(64, padding=8),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							get_norm_trans(args),
						]))

		test_dataset = ImageFolderDataset("/mnt/ext/zhicong/VGGface2/test_cls_subset", 
						preprocess_before = transforms.Compose([
							transforms.Resize(84),
							transforms.CenterCrop(64),
						]),
					   transform=transforms.Compose([
							transforms.ToTensor(),
							get_norm_trans(args),
						]))


	elif args.dataset.lower()=='mini-imagenet':
		train_dataset = ImageFolderDataset("/mnt/ext/zhicong/mini-imagenet/train", 
						preprocess_before = transforms.Compose([
							transforms.Resize(64),
						]),
					   transform=transforms.Compose([
							transforms.RandomCrop(64, padding=8),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							get_norm_trans(args),
						]))

		test_dataset = ImageFolderDataset("/mnt/ext/zhicong/mini-imagenet/test", 
						preprocess_before = transforms.Compose([
							transforms.Resize(64),
						]),
					   transform=transforms.Compose([
							transforms.ToTensor(),
							get_norm_trans(args),
						]))


	elif args.dataset.lower() == 'imagenet':
		train_dataset = None # not required
		test_dataset = datasets.ImageFolder("/mnt/ext/ImageNet/val", 
					  transform=transforms.Compose([
							transforms.Resize(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							get_norm_trans(args),
						]))

	return train_dataset, test_dataset




