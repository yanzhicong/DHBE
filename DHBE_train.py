import os
import sys
import random
import argparse
import pickle as pkl
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim

import gan
import models
from dataloader import get_dataset, get_norm_trans, get_norm_trans_inv
import my_utils as utils

from backdoor.base import SimpleSubset, TriggerPastedTestDataset

import vis



def train(args, teacher, student, generator, pert_generator,
	norm_trans, norm_trans_inv, 
	optimizer, epoch, plotter=None, 
	):
	teacher.eval()
	student.train()
	generator.train()
	pert_generator.train()


	optimizer_S, optimizer_G, optimizer_Gp = optimizer

	for i in range( args.epoch_itrs ):
		for k in range(5):
			z = torch.randn((args.batch_size, args.nz)).cuda()
			z2 = torch.randn((args.batch_size, args.nz2)).cuda()

			optimizer_S.zero_grad()

			fake_data = generator(z).detach()
			pert = pert_generator(z2)
			pert = pert_generator.random_pad(pert).detach()

			fake_img = norm_trans_inv(fake_data)
			patched_img = fake_img + pert
			patched_data = norm_trans(patched_img)

			t_logit = teacher(fake_data)
			s_logit = student(fake_data)

			s_logit_pret = student(patched_data)

			loss_S1 = F.l1_loss(s_logit, t_logit.detach())
			loss_S2 = F.l1_loss(s_logit_pret, s_logit.detach())			
			loss_S = loss_S1 + loss_S2 * args.loss_weight_d1

			loss_S.backward()
			optimizer_S.step()

		z = torch.randn((args.batch_size, args.nz)).cuda()
		optimizer_G.zero_grad()
		fake_data = generator(z)
		t_logit = teacher(fake_data)
		s_logit = student(fake_data)

		loss_G1 = - F.l1_loss( s_logit, t_logit ) 
		loss_G2 = utils.get_image_prior_losses_l1(fake_data)
		loss_G3 = utils.get_image_prior_losses_l2(fake_data)
		loss_G = loss_G1 + args.loss_weight_tvl1 * loss_G2 + args.loss_weight_tvl2 * loss_G3


		loss_G.backward()
		optimizer_G.step()

		z = torch.randn((args.batch_size, args.nz)).cuda()
		z2 = torch.randn((args.batch_size, args.nz2)).cuda()

		optimizer_Gp.zero_grad()

		fake_data = generator(z).detach()
		pert = pert_generator(z2)
		pert = pert_generator.random_pad(pert)
		fake_img = norm_trans_inv(fake_data)
		patched_img = fake_img + pert
		patched_data = norm_trans(patched_img)

		s_logit = student(fake_data)
		s_logit_pret = student(patched_data)

		loss_Gp = - F.l1_loss(s_logit, s_logit_pret)

		loss_Gp.backward()
		optimizer_Gp.step()


		if i % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
				epoch, i, args.epoch_itrs, 100*float(i)/float(args.epoch_itrs), loss_G.item(), loss_S.item()))

			if plotter is not None:
				plotter.scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
				plotter.scalar('Loss_S1', (epoch-1)*args.epoch_itrs+i, loss_S1.item())
				plotter.scalar('Loss_S2', (epoch-1)*args.epoch_itrs+i, loss_S2.item())

				plotter.scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())
				plotter.scalar('Loss_G_l1', (epoch-1)*args.epoch_itrs+i, loss_G1.item())
				plotter.scalar('Loss_G_tvl1', (epoch-1)*args.epoch_itrs+i, loss_G2.item())
				plotter.scalar('Loss_G_tvl2', (epoch-1)*args.epoch_itrs+i, loss_G3.item())

				plotter.scalar('Loss_Gp', (epoch-1)*args.epoch_itrs+i, loss_Gp.item())




def main():
	# Training settings
	parser = argparse.ArgumentParser(description='DHBE CIFAR')
	parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
	parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
	parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 500)')
	parser.add_argument('--epoch_itrs', type=int, default=50)

	parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
	parser.add_argument('--lr_G', type=float, default=1e-3, help='learning rate (default: 0.1)')
	parser.add_argument('--lr_decay', type=float, default=0.1)
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

	parser.add_argument('--input_dir', type=str, default="train_teacher_badnets_cifar10_resnet18_e_200_tri1_3x3_t9_0_0_n300_results")
	parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'vggface2_subset', 'mini-imagenet'], help='dataset name (default: mnist)')
	
	parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
	parser.add_argument('--nz', type=int, default=256)

	parser.add_argument('--loss_weight_tvl1', type=float, default=0.0)
	parser.add_argument('--loss_weight_tvl2', type=float, default=0.0001)

	parser.add_argument('--lr_Gp', type=float, default=1e-3, help='learning rate (default: 0.1)')
	parser.add_argument('--loss_weight_d1', type=float, default=0.1)
	parser.add_argument('--patch_size', type=int, default=5)
	parser.add_argument('--nz2', type=int, default=256)

	parser.add_argument('--vis_generator', action='store_true', default=False)
	
	args = parser.parse_args()
	args.num_classes = {"cifar10":10, "cifar100":100, "mnist":10, "vggface2_subset":100, "svhn":10, "mini-imagenet":100}.get(args.dataset, 10)
	args.img_size = {"cifar10":32, "cifar100":32, "mnist":28, "vggface2_subset":64, "svhn":32, "mini-imagenet":64}.get(args.dataset, 32)
	args.img_channels = {"cifar10":3, "cifar100":3, "mnist":1, "vggface2_subset":3, "svhn":3, "mini-imagenet":3}.get(args.dataset, 3)
	
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	print(args)

	param_string = "e_{}_ps_{}_wd1_{}_lwtvl1_{}_lwtvl2_{}_lrs_{}_lrg_{}_lrgp_{}_nz_{}_nz2_{}"
	param_string = param_string.format(args.epochs, args.patch_size, args.loss_weight_d1, args.loss_weight_tvl1, args.loss_weight_tvl2, args.lr_S, args.lr_G, args.lr_Gp, args.nz, args.nz2)

	output_dir = os.path.join(args.input_dir, __file__.split('.')[0] + "_{}_results".format(param_string))

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	os.makedirs(os.path.join(output_dir, "student"), exist_ok=True)
	os.makedirs(os.path.join(output_dir, "generator"), exist_ok=True)
	if args.vis_generator:
		os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
	if args.vis_sensitivity:
		os.makedirs(os.path.join(output_dir, "sensitivity"), exist_ok=True)
	if args.save_checkpoint:
		os.makedirs(os.path.join(output_dir, "train_stats"), exist_ok=True)

	norm_trans = get_norm_trans(args)
	norm_trans_inv = get_norm_trans_inv(args)
	train_ds, test_ds = get_dataset(args)

	trigger, target_class = utils.infer_trigger_from_path(args.input_dir, train_ds, args.img_size)
	args.model = utils.infer_model_from_path(args.input_dir)

	# For testing ASR and ACC
	backdoored_test_ds = TriggerPastedTestDataset(test_ds, trigger, target_class=target_class)

	ckpt_path = os.path.join(args.input_dir, "teacher", "{}-{}.pt".format(args.dataset, args.model))
	teacher = models.get_model(args)
	student = models.get_model(args)
	generator = gan.GeneratorB(nz=args.nz, nc=args.img_channels, img_size=args.img_size)

	pert_generator = gan.PatchGeneratorPreBN(nz=args.nz2, nc=args.img_channels, patch_size=args.patch_size, out_size=args.img_size)


	teacher.load_state_dict(torch.load(ckpt_path))
	print("Teacher restored from %s"%(ckpt_path))
	student.load_state_dict(torch.load(ckpt_path))
	print("Student restored from %s"%(ckpt_path))


	teacher = teacher.cuda()
	student = student.cuda()
	generator = generator.cuda()
	pert_generator = pert_generator.cuda()

	teacher.eval()

	optimizer_S = optim.SGD( student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )
	optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
	optimizer_Gp = optim.Adam( pert_generator.parameters(), lr=args.lr_Gp )

	lr_decay_steps = [0.6, 0.8]
	lr_decay_steps = [int(e * args.epochs) for e in lr_decay_steps]

	scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, lr_decay_steps, args.lr_decay)
	scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, lr_decay_steps, args.lr_decay)
	scheduler_Gp = optim.lr_scheduler.MultiStepLR(optimizer_Gp, lr_decay_steps, args.lr_decay)

	plotter = vis.Plotter()


	for epoch in range(1, args.epochs + 1):
		# Train
		
		if epoch == 1:
			acc, asr = utils.test_model_acc_and_asr(args, student, backdoored_test_ds)
			plotter.scalar("test_acc", 0, acc)
			plotter.scalar("test_asr", 0, asr)

		train(args, teacher=teacher, student=student, generator=generator, 
					pert_generator=pert_generator,
					norm_trans=norm_trans, norm_trans_inv=norm_trans_inv,
					optimizer=[optimizer_S, optimizer_G, 
					optimizer_Gp
					], epoch=epoch, plotter=plotter, 
					# trigger=trigger
					)

		scheduler_S.step()
		scheduler_G.step()
		scheduler_Gp.step()

		# Test
		if args.vis_generator:
			utils.test_generators(args, {'img':generator}, args.nz, epoch, output_dir, plotter, norm_trans_inv=norm_trans_inv)
			utils.test_generators(args, {'pert':pert_generator}, args.nz2, epoch, output_dir, plotter, norm_trans_inv=lambda x:(x+1.0)/2.0)
		
		acc, asr = utils.test_model_acc_and_asr(args, student, backdoored_test_ds)
		print("-"*30+"\n"+"Epoch {}: acc : {:.3f}, asr : {:.3f}".format(epoch, acc, asr))
		plotter.scalar("test_acc", epoch, acc)
		plotter.scalar("test_asr", epoch, asr)


		if epoch % 10 == 0 or epoch == args.epochs:
			torch.save(student.state_dict(), os.path.join(output_dir, "student/%s-%s.pt"%(args.dataset, args.model)))
			torch.save(generator.state_dict(), os.path.join(output_dir, "generator/%s-%s-generator.pt"%(args.dataset, args.model)))
			torch.save(pert_generator.state_dict(), os.path.join(output_dir, "generator/%s-%s-pert_generator.pt"%(args.dataset, args.model)))


		plotter.to_csv(output_dir)
		plotter.to_html_report(os.path.join(output_dir, "index.html"))




if __name__ == '__main__':
	main()


