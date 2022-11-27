
import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import models
from dataloader import get_dataset



import vis


from backdoor.trigger import Trigger, cifar_triggers, mnist_triggers
from backdoor.base import TriggerPastedTestDataset
from backdoor.dirty_label import DirtyLabelPoisonedDataset

import my_utils as utils


def train(args, model, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
		data, target = data.cuda(), target.cuda()
		optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		if args.verbose and batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def main():
	# Training settings
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
	parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'vggface2_subset', 'mini-imagenet'], help='dataset name (default: mnist)')
	parser.add_argument('--model', type=str, default='resnet18', help='model name (default: mnist)')
	parser.add_argument('--trigger_name', type=str, default='tri1_3x3')
	parser.add_argument('--target_class', type=int, default=-1)
	parser.add_argument('--num_poison_images', type=int, default=300)
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
	parser.add_argument('--verbose', action='store_true', default=True)
	parser.add_argument('--trigger_offset', type=int, default=0)

	args = parser.parse_args()


	args.num_classes = {"cifar10":10, "cifar100":100, "mnist":10, "vggface2_subset":100, "svhn":10, "mini-imagenet":100}.get(args.dataset, 10)
	args.img_channels = {"cifar10":3, "cifar100":3, "mnist":1, "vggface2_subset":3, "svhn":3, "mini-imagenet":3}.get(args.dataset, 3)

	if args.target_class == -1:
		target_class = args.num_classes-1
	else:
		target_class = args.target_class

	num_poison_images = args.num_poison_images
	trigger_name = "{}_t{}_{}_{}".format(args.trigger_name, target_class, args.trigger_offset, args.trigger_offset)

	param_string = "{}_{}_e_{}_{}_n{}"
	param_string = param_string.format(args.dataset, args.model, args.epochs, trigger_name, num_poison_images)
	if args.seed != 1:
		param_string += "_seed_{}".format(args.seed)

	output_dir = __file__.split('.')[0] + "_{}_results".format(param_string)

	os.makedirs(os.path.join(output_dir, 'teacher'), exist_ok=True)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	print(args)

	train_ds, test_ds = get_dataset(args)

	trigger = Trigger(args.trigger_name, cifar_triggers[args.trigger_name] if args.img_channels == 3 else mnist_triggers[args.trigger_name],
			offset_to_right=args.trigger_offset,
			offset_to_bottom=args.trigger_offset)

	train_ds = DirtyLabelPoisonedDataset(train_ds, trigger, target_class=target_class, 
				num_poison_images=num_poison_images, 
				sample_save_path=os.path.join(output_dir, 'poisoned_samples.png'))
	test_ds = TriggerPastedTestDataset(test_ds, trigger, target_class=target_class)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

	model = models.get_model(args)
	model = model.cuda()

	steps = [0.5, 0.8, 0.9]
	steps = [int(s * args.epochs) for s in steps]

	optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, 0.1)

	plotter = vis.Plotter()
	best_acc = 0

	for epoch in range(1, args.epochs + 1):
		
		train(args, model, train_loader, optimizer, epoch)

		acc, asr = utils.test_model_acc_and_asr(args, model, test_ds)

		print("\t acc : {}, asr : {}".format(acc, asr))

		plotter.scalar("test_acc", epoch+1, acc)
		plotter.scalar("test_asr", epoch+1, asr)
		plotter.scalar("lr", epoch+1, scheduler.get_last_lr()[0])

		plotter.to_csv(output_dir)
		plotter.to_html_report(os.path.join(output_dir, "index.html"))

		if acc > best_acc:
			best_acc = acc
			torch.save(model.state_dict(), os.path.join(output_dir, "teacher/%s-%s.pt"%(args.dataset, args.model)))

		scheduler.step()


	print("Best Acc=%.6f"%best_acc)





if __name__ == '__main__':
	main()
