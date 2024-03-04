import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, DA_loss, patchnce, MINE
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp,PatchSampleF,get_keys
from models.stylegan2.model import  Discriminator
from training.ranger import Ranger
import itertools
import numpy
# from stylegan2_ada import *
# import dnnlib
# import legacy
import numpy as np
import random
from models.stylegan2.model import PixelNorm,EqualLinear
from models.networks import StyleEncoder
from training.gp import *
import functools
# from torch_ema import ExponentialMovingAverage


def requires_grad(model, flag=True, target_layer=None):
	# print(target_layer)
	for name, param  in model.named_parameters():
		if target_layer is None or target_layer in name:
			# print(name)
			param.requires_grad = flag

class StatisticNetwork_X2c1(nn.Module): #x:3*256*256 c1:128*64*64
    def __init__(self):
        super(StatisticNetwork_X2c1, self).__init__()

        # self.frame_encoder_w = FanFusion(256)
        self.net1 = nn.Sequential(
			nn.Conv2d(3,4,4,4,0),#4*64*64
			nn.Conv2d(4,8,4,4,0),#8*16*16
			nn.Conv2d(8,16,4,4,0) #16*4*4
			)  # 256
        self.net2 = nn.Sequential(
			nn.Conv2d(128,64,2,2,0), #64*32*32
			nn.Conv2d(64,32,2,2,0), #32*16*16
			nn.Conv2d(32,16,4,4,0) # 16*4*4
			) #256  
        self.fusion_net = nn.Sequential(
            nn.Linear(256*2, 1),
        )

    def forward(self, image, mfcc):
        y1 = self.net1(image).view(-1, 256)
        x1 = self.net2(mfcc).view(-1, 256)
        cat_feature = torch.cat((x1, y1), dim=1)  # fusion use cat is stable than element-wise plus
        return self.fusion_net(cat_feature)

class StatisticNetwork_X2c3(nn.Module): #x:3*256*256 c3:512*16*16
    def __init__(self):
        super(StatisticNetwork_X2c3, self).__init__()

        # self.frame_encoder_w = FanFusion(256)
        self.net1 = nn.Sequential(
			nn.Conv2d(3,4,4,4,0),#4*64*64
			nn.Conv2d(4,8,4,4,0),#8*16*16
			nn.Conv2d(8,16,4,4,0)#16*4*4
			)  # 256
        self.net2 = nn.Sequential(
			nn.Conv2d(512,128,2,2,0),#128*8*8
			nn.Conv2d(128,64,2,2,0),#64*4*4
			nn.Conv2d(64,16,1,1,0) #16*4*4
			)  # 256
        self.fusion_net = nn.Sequential(
            nn.Linear(256*2, 1),
        )

    def forward(self, mfcc, image):
        x1 = self.net1(mfcc).view(-1, 256)
        y1 = self.net2(image).view(-1, 256)
        cat_feature = torch.cat((x1, y1), dim=1)  # fusion use cat is stable than element-wise plus
        return self.fusion_net(cat_feature)

class RotationPreNet(nn.Module):
	def __init__(self,class_num=4):
		super(RotationPreNet, self).__init__()
		self.ReduceNet = nn.Sequential(
			nn.Conv2d(512,128,2,2,0),#128*8*8
			nn.ReLU(),
			nn.Conv2d(128,64,2,2,0),#64*4*4
			nn.ReLU(),
			nn.Conv2d(64,16,1,1,0) #16*4*4
			)  # 256
		self.classifier = nn.Sequential(
			nn.Linear(256, 128),
			nn.Linear(128, class_num)
		)
	
	def forward(self, x1):
		input_cat = x1
		# print(input_cat.size())
		# raise RuntimeError
		feat = self.ReduceNet(input_cat)
		pred = self.classifier(feat.view(-1,256))
		return pred
"""class Discriminator(nn.Module):
    def __init__(self, spectral_normed, num_rotation,
                ssup, channel, resnet = False):
        super(Discriminator, self).__init__()
        self.resnet = resnet
        self.num_rotation = num_rotation
        self.ssup = ssup

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv1 = conv2d(channel, 64, kernel_size = 3, stride = 1, padding = 1,
                            spectral_normed = spectral_normed)
        self.conv2 = conv2d(64, 128, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv3 = conv2d(128, 256, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv4 = conv2d(256, 512, spectral_normed = spectral_normed,
                            padding = 0)
        self.fully_connect_gan1 = nn.Linear(512, 1)
        self.fully_connect_rot1 = nn.Linear(512, 4)
        self.softmax = nn.Softmax()

        self.re1 = Residual_D(channel, 128, spectral_normed = spectral_normed,
                            down_sampling = True, is_start = True)
        self.re2 = Residual_D(128, 128, spectral_normed = spectral_normed,
                            down_sampling = True)
        self.re3 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.re4 = Residual_D(128, 128, spectral_normed = spectral_normed)
        self.fully_connect_gan2 = nn.Linear(128, 1)
        self.fully_connect_rot2 = nn.Linear(128, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.resnet == False:
            conv1 = self.lrelu(self.conv1(x))
            conv2 = self.lrelu(self.conv2(conv1))
            conv3 = self.lrelu(self.conv3(conv2))
            conv4 = self.lrelu(self.conv4(conv3))
            conv4 = torch.view(conv4.size(0)*self.num_rotation, -1)
            gan_logits = self.fully_connect_gan1(conv4)
            if self.ssup:
                rot_logits = self.fully_connect_rot1(conv4)
                rot_prob = self.softmax(rot_logits)
        else:
            re1 = self.re1(x)
            re2 = self.re2(re1)
            re3 = self.re3(re2)
            re4 = self.re4(re3)
            re4 = self.relu(re4)
            re4 = torch.sum(re4,dim = (2,3))
            gan_logits = self.fully_connect_gan2(re4)
            if self.ssup:
                rot_logits = self.fully_connect_rot2(re4)
                rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.sigmoid(gan_logits), gan_logits"""

class Z_mapping(nn.Module):
	def __init__(self,n_mlp=8,style_dim=512,code_dim=6):
		super(Z_mapping, self).__init__()
		layers = [PixelNorm()]
		for i in range(n_mlp):
			layers.append(
				EqualLinear(
					style_dim, style_dim, activation='fused_lrelu'
				)
			)
		layers.append(
			EqualLinear(
					style_dim, style_dim*code_dim, activation='fused_lrelu'
				)
		)
		# self.code_dim=code_dim
		self.style=nn.Sequential(*layers)
	def forward(self,s):
		# [self.style(s) for s in styles]
		style=self.style(s)
		
		# print(style.size())
		return style.view(style.size(0),-1,512)

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device
		
		# if(self.opts.DA):
		# 	ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
		# 	self.opts = ckpt['opts']
		# Initialize network
		self.net = pSp(self.opts).to(self.device)
		if(self.opts.withNCE):
			self.netF = PatchSampleF()
		if((self.opts.GAN_lambda!=0) or (self.opts.D_lambda!=0)):
			self.Discriminator = Discriminator(1024).to(self.device)
			self.load_D_weights()
		if(self.opts.SS!=0):
			self.RotateNet = RotationPreNet().to(self.device)
			if(self.opts.DA):
				self.load_SS_weights()
		if((self.opts.remapping!=0) or (self.opts.Encv2!=0)):
			self.Znet=Z_mapping(code_dim=self.opts.Zlevel).to(self.device)
			self.load_Z_weights()
		if(self.opts.StyleEnc!=0):
			self.StyleNet=StyleEncoder().to(self.device)
		if(self.opts.EMA!=0):
			self.ema = ExponentialMovingAverage(self.Discriminator.parameters(), decay=0.995)
			
		if((self.opts.TexPr_lambda !=0) or (self.opts.fakeY)):
			self.Teacher=pSp(self.opts).to(self.device)
		if(self.opts.MI!=0):
			self.StatisticNet_X2Z=StatisticNetwork_X2c1().to(self.device)
			self.StatisticNet_Z2Y=StatisticNetwork_X2c3().to(self.device)
			self.mine_x2c1 = MINE.MINE(self.StatisticNet_X2Z)
			self.mine_x2c3 = MINE.MINE(self.StatisticNet_Z2Y)
		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.t_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()
		if(self.opts.use_ada):
			network_pkl='./metfaces.pkl'
			print('Loading networks from "%s"...' % network_pkl)
			with dnnlib.util.open_url(network_pkl) as f:
				self.ada_GAN = legacy.load_network_pkl(f)['G_ema'].to(self.device)

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()
		if (self.opts.withNCE):
			# self.PatchNCELoss = patchnce.PatchNCELoss().to(self.device).eval()
			self.criterionNCE = []
			for i in range(3):
				self.criterionNCE.append(patchnce.PatchNCELoss(self.opts).to(self.device))
		if(self.opts.KD):
			self.da_loss = DA_loss.DAloss().to(self.device).eval()
		# self.Rot=torchvision.transforms.functional.rotate

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()
		if(self.opts.GAN_lambda!=0):
			self.optim_D = self.configure_optim_D()
		if(self.opts.SS!=0 and (not self.opts.DA)):
			self.optim_SS = self.configure_optim_SS()
		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps
	def d_logistic_loss(real_pred, fake_pred):
		real_loss = F.softplus(-real_pred)
		fake_loss = F.softplus(fake_pred)

		return real_loss.mean() + fake_loss.mean()
	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				if(self.opts.freezeE):
					requires_grad(self.net.encoder, False, target_layer=f'input_layer')
					if(self.opts.Notadd):
						for i in range(21):
							requires_grad(self.net.encoder, False, target_layer=f'body.{i}')
					else:
						requires_grad(self.net.encoder, False, target_layer=f'body')
					for i in range(3,18):
						requires_grad(self.net.encoder, False, target_layer=f'styles.{i}')
				# raise RuntimeError

				self.optimizer.zero_grad()
				if(self.opts.SS!=0):
					x, GT, x1, x2, x3 = batch
					x, GT, x1, x2, x3 = x.to(self.device).float(), GT.to(self.device).float(), x1.to(self.device).float(), x2.to(self.device).float(), x3.to(self.device).float()
					y=x
				else:
					x, GT = batch
					x, GT = x.to(self.device).float(), GT.to(self.device).float()
					y=GT
				y_hat, latent = self.net.forward(x, return_latents=True)
				if(self.opts.fakeY):
					with torch.no_grad():
						y = self.Teacher.forward(x)
				if(self.opts.SS!=0):
					loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, GT,x1=x1,x2=x2,x3=x3)
				else:
					loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, GT)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
	def D_train(self):
		self.net.train()
		N_CRITIC=1
		self.Discriminator.train()
		if(self.opts.MI!=0):
			self.StatisticNet_X2Z.train()
			self.StatisticNet_Z2Y.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				if(self.opts.SS!=0):
					x, GT, x1, x2, x3 = batch
					x, GT, x1, x2, x3 = x.to(self.device).float(), GT.to(self.device).float(), x1.to(self.device).float(), x2.to(self.device).float(), x3.to(self.device).float()
					y=x
				else:
					x, GT = batch
					x, GT = x.to(self.device).float(), GT.to(self.device).float()
					y=x
				requires_grad(self.net, True)
				requires_grad(self.Discriminator, False)
				if(self.opts.SS!=0):
					requires_grad(self.RotateNet,False)
				if(self.opts.fakeY):
					# with torch.no_grad():
					if(self.opts.MultiFakeY!=0):
						w,h = x.size()[2], x.size()[3]
						m = random.randint(0,w-(w//4)-1)
						n = random.randint(0,h-(h//4)-1)
						x_=x[:,:,m:m+(w//4),n:n+(h//4)]
						# print(x.size(),x_.size())
						x_=F.interpolate(x_,scale_factor=4)
						fake_real=self.Teacher.forward(x_)
						# fake_Real2=self.Teacher.forward(x)
						# self.parse_and_log_images(None,fake_Real,fake_Real2,x_,title='images')
						# raise RuntimeError
					else:
						if(self.opts.GfromZ!=0):
							z=torch.randn(x.size()[0],6,512).float().cuda()
							fake_real, _ = self.Teacher.forward(x, return_latents=True,latentz=z)
						elif(self.opts.remapping):
							z=torch.randn(x.size()[0],512).float().cuda()
							z=self.Znet(z)
							# raise RuntimeError
							fake_real, _  = self.net.forward(x, return_latents=True,latentz=z)
						else:
							fake_real = self.Teacher.forward(x)
				elif(self.opts.use_ada): 
					with torch.no_grad():
						label = torch.zeros([self.opts.batch_size, self.ada_GAN.c_dim], device=self.device)
						seed=np.random.randint(1,1500,[self.opts.batch_size])
						z = torch.from_numpy(np.random.RandomState(seed).randn(self.opts.batch_size, self.ada_GAN.z_dim)).to(self.device)
						print(z.size())
						fake_real,feat = self.ada_GAN(z, label, truncation_psi=1.0, noise_mode='const',return_features=True)
						raise RuntimeError
						# self.parse_and_log_images(id_logs, x, y, fake_real, title='images')
						# raise RuntimeError
						# y=fake_real
				if(batch_idx % N_CRITIC ==0):
					self.optimizer.zero_grad()
					# x, y = batch
					# x, y = x.to(self.device).float(), y.to(self.device).float()
					
					if(self.opts.GfromZ!=0):
						z=torch.randn(x.size()[0],self.opts.Zlevel,512).float().cuda()
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					elif(self.opts.remapping!=0):
						z=torch.randn(x.size()[0],512).float().cuda()
						z=self.Znet(z)
						# z= z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
						# print(z.size())
						# raise RuntimeError
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					elif(self.opts.StyleEnc!=0):
						z=torch.randn(x.size()[0],512).float().cuda()
						# print(z.size())
						z=z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					else:
						y_hat, latent = self.net.forward(x, return_latents=True)
					# print(y_hat.size(),x.size())
					# print(latent.size())
					if(self.opts.SS!=0):
						loss, loss_dict, id_logs = self.calc_loss(x, x, y_hat, latent, GT,x1=x1,x2=x2,x3=x3)
					else:
						loss, loss_dict, id_logs = self.calc_loss(x, x, y_hat, latent, GT)
					loss.backward()
					self.optimizer.step()
				
				requires_grad(self.net, False)
				requires_grad(self.Discriminator, True)
				self.optim_D.zero_grad()
				if(self.opts.GfromZ!=0):
					z=torch.randn(x.size()[0],self.opts.Zlevel,512).float().cuda()
					y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
				elif(self.opts.remapping!=0):
					z=torch.randn(x.size()[0],512).float().cuda()
					z=self.Znet(z)
					# z=z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
					# raise RuntimeError
					y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
				elif(self.opts.StyleEnc!=0):
					# z=self.StyleNet(x).view(x.size(0),-1)
					z=torch.randn(x.size()[0],512).float().cuda()
					# print(z.size())
					z=z.unsqueeze(1).repeat(1, 6, 1)
					y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
				else:
					y_hat, latent = self.net.forward(x, return_latents=True)
				
				fake_out = self.Discriminator(y_hat)
				if(self.opts.fakeY or self.opts.use_ada):
					real_out=self.Discriminator(fake_real)
					gp = gradient_penalty(functools.partial(self.Discriminator), fake_real, y_hat.detach(), gp_mode='lp')
				else:
					real_out = self.Discriminator(GT)
					gp = gradient_penalty(functools.partial(self.Discriminator), GT, y_hat.detach(), gp_mode='lp')

				
				if(self.opts.Encv2!=0):
					z=torch.randn(x.size()[0],512).float().cuda()
					z=self.Znet(z)
					y2_hat, _ = self.net.forward(x, return_latents=True,latentz=z)
					fake_out2 = self.Discriminator(y2_hat)
				# raise RuntimeError

				real_loss = F.softplus(-real_out).mean()
				fake_loss = F.softplus(fake_out).mean()
				# real_loss = -real_out.mean()
				# fake_loss = fake_out.mean()

				if(self.opts.Encv2!=0):
					fake_loss+=F.softplus(fake_out2).mean()
				loss_D = real_loss + fake_loss + gp * 10
				if(self.opts.EMA!=0):
					loss_D+=RLC
				loss_dict['D_loss']=float(loss_D)

				loss_D.backward()
				self.optim_D.step()
				if(self.opts.EMA!=0):
					self.ema.update()
				"""if(self.opts.SS!=0):
					requires_grad(self.net, False)
					requires_grad(self.Discriminator, False)
					requires_grad(self.RotateNet,True)
					self.optim_SS.zero_grad()
					x1=self.Rot(x,90)
					x2=self.Rot(x,180)
					x3=self.Rot(x,270)
					_,_,xf=self.net.intermediate_encode(x)
					_,_,x1f=self.net.intermediate_encode(x1)
					_,_,x2f=self.net.intermediate_encode(x2)
					_,_,x3f=self.net.intermediate_encode(x3)
					pred0=self.RotateNet(xf,xf)
					pred1=self.RotateNet(xf,x1f)
					pred2=self.RotateNet(xf,x2f)
					pred3=self.RotateNet(xf,x3f)
					pred4=self.RotateNet(x1f,x1f)
					pred5=self.RotateNet(x1f,x2f)
					pred6=self.RotateNet(x1f,x3f)
					pred7=self.RotateNet(x2f,x2f)
					pred8=self.RotateNet(x2f,x3f)
					pred9=self.RotateNet(x3f,x3f)
					SS_loss = nn.CrossEntropyLoss()(pred0,0) + nn.CrossEntropyLoss()(pred0,1) + nn.CrossEntropyLoss()(pred0,2) + nn.CrossEntropyLoss()(pred0,3) + \
							nn.CrossEntropyLoss()(pred0,4) + nn.CrossEntropyLoss()(pred0,5) + nn.CrossEntropyLoss()(pred0,6) + \
							nn.CrossEntropyLoss()(pred0,7) + nn.CrossEntropyLoss()(pred0,8) + nn.CrossEntropyLoss()(pred0,9)
					loss_dict['SS_Dloss']=float(SS_loss)
					self.optim_SS.step()"""

				# if(batch_idx+1>=N_CRITIC):
				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 25 == 0):

					if(self.opts.DA and (self.opts.oneshot!=0) and (self.opts.GfromZ!=0)):
						self.parse_and_log_images_MULTI(id_logs, x, GT, y_hat, title='images/train/faces')
					elif(self.opts.Encv2!=0):
						with torch.no_grad():
							z=torch.randn(x.size(0),512).cuda()
							z=self.Znet(z)
							y_random,_ = self.net.forward(x, return_latents=True,latentz=z)
							self.parse_and_log_images(id_logs, x, y_random, y_hat, title='images/train/faces')
					elif(self.opts.fakeY):
						# print('11',fake_real.size())
						self.parse_and_log_images(id_logs, x, fake_real, y_hat, title='images/train/faces')
					else:
						self.parse_and_log_images(id_logs, x, GT, y_hat, title='images/train/faces')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break
				self.global_step += 1
	def Test_train(self):
		self.net.train()
		N_CRITIC=1
		self.Discriminator.train()
		if(self.opts.MI!=0):
			self.StatisticNet_X2Z.train()
			self.StatisticNet_Z2Y.train()
		
		while self.global_step < self.opts.max_steps:
			for batch_idx0, batch in enumerate(self.train_dataloader):
				self.load_D_weights()
				self.net = pSp(self.opts).to(self.device)
				if(self.opts.SS!=0):
					x, GT, x1, x2, x3 = batch
					x, GT, x1, x2, x3 = x.to(self.device).float(), GT.to(self.device).float(), x1.to(self.device).float(), x2.to(self.device).float(), x3.to(self.device).float()
					y=x
				else:
					x, GT = batch
					x, GT = x.to(self.device).float(), GT.to(self.device).float()
					y=x
				for batch_idx in range(76):
					
					requires_grad(self.net, True)
					requires_grad(self.Discriminator, False)
					if(self.opts.SS!=0):
						requires_grad(self.RotateNet,False)
					if(self.opts.fakeY):
						# with torch.no_grad():
						if(self.opts.MultiFakeY!=0):
							w,h = x.size()[2], x.size()[3]
							m = random.randint(0,w-(w//4)-1)
							n = random.randint(0,h-(h//4)-1)
							x_=x[:,:,m:m+(w//4),n:n+(h//4)]
							# print(x.size(),x_.size())
							x_=F.interpolate(x_,scale_factor=4)
							fake_real=self.Teacher.forward(x_)
							# fake_Real2=self.Teacher.forward(x)
							# self.parse_and_log_images(None,fake_Real,fake_Real2,x_,title='images')
							# raise RuntimeError
						else:
							if(self.opts.GfromZ!=0):
								z=torch.randn(x.size()[0],6,512).float().cuda()
								fake_real, _ = self.Teacher.forward(x, return_latents=True,latentz=z)
							elif(self.opts.remapping):
								z=torch.randn(x.size()[0],512).float().cuda()
								z=self.Znet(z)
								# raise RuntimeError
								fake_real, _  = self.net.forward(x, return_latents=True,latentz=z)
							else:
								fake_real = self.Teacher.forward(x)
					if(batch_idx % N_CRITIC ==0):
						self.optimizer.zero_grad()
						# x, y = batch
						# x, y = x.to(self.device).float(), y.to(self.device).float()
						
						if(self.opts.GfromZ!=0):
							z=torch.randn(x.size()[0],self.opts.Zlevel,512).float().cuda()
							y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
						elif(self.opts.remapping!=0):
							z=torch.randn(x.size()[0],512).float().cuda()
							z=self.Znet(z)
							# z= z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
							# print(z.size())
							# raise RuntimeError
							y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
						elif(self.opts.StyleEnc!=0):
							z=torch.randn(x.size()[0],512).float().cuda()
							# print(z.size())
							z=z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
							y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
						else:
							y_hat, latent = self.net.forward(x, return_latents=True)
						# print(y_hat.size(),x.size())
						# print(latent.size())
						loss, loss_dict, id_logs = self.calc_loss(x, x, y_hat, latent, GT)
						loss.backward()
						self.optimizer.step()
					
					requires_grad(self.net, False)
					requires_grad(self.Discriminator, True)
					self.optim_D.zero_grad()
					if(self.opts.GfromZ!=0):
						z=torch.randn(x.size()[0],self.opts.Zlevel,512).float().cuda()
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					elif(self.opts.remapping!=0):
						z=torch.randn(x.size()[0],512).float().cuda()
						z=self.Znet(z)
						# z=z.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
						# raise RuntimeError
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					elif(self.opts.StyleEnc!=0):
						# z=self.StyleNet(x).view(x.size(0),-1)
						z=torch.randn(x.size()[0],512).float().cuda()
						# print(z.size())
						z=z.unsqueeze(1).repeat(1, 6, 1)
						y_hat, latent = self.net.forward(x, return_latents=True,latentz=z)
					else:
						y_hat, latent = self.net.forward(x, return_latents=True)
					
					fake_out = self.Discriminator(y_hat)
					if(self.opts.fakeY or self.opts.use_ada):
						real_out=self.Discriminator(fake_real)
						gp = gradient_penalty(functools.partial(self.Discriminator), fake_real, y_hat.detach(), gp_mode='lp')
					else:
						real_out = self.Discriminator(GT)
						gp = gradient_penalty(functools.partial(self.Discriminator), GT, y_hat.detach(), gp_mode='lp')

					
					if(self.opts.Encv2!=0):
						z=torch.randn(x.size()[0],512).float().cuda()
						z=self.Znet(z)
						y2_hat, _ = self.net.forward(x, return_latents=True,latentz=z)
						fake_out2 = self.Discriminator(y2_hat)
					# raise RuntimeError

					real_loss = F.softplus(-real_out).mean()
					fake_loss = F.softplus(fake_out).mean()
					# real_loss = -real_out.mean()
					# fake_loss = fake_out.mean()

					if(self.opts.Encv2!=0):
						fake_loss+=F.softplus(fake_out2).mean()
					loss_D = real_loss + fake_loss + gp * 10
					if(self.opts.EMA!=0):
						loss_D+=RLC
					loss_dict['D_loss']=float(loss_D)

					loss_D.backward()
					self.optim_D.step()
					if(self.opts.EMA!=0):
						self.ema.update()
					"""if(self.opts.SS!=0):
						requires_grad(self.net, False)
						requires_grad(self.Discriminator, False)
						requires_grad(self.RotateNet,True)
						self.optim_SS.zero_grad()
						x1=self.Rot(x,90)
						x2=self.Rot(x,180)
						x3=self.Rot(x,270)
						_,_,xf=self.net.intermediate_encode(x)
						_,_,x1f=self.net.intermediate_encode(x1)
						_,_,x2f=self.net.intermediate_encode(x2)
						_,_,x3f=self.net.intermediate_encode(x3)
						pred0=self.RotateNet(xf,xf)
						pred1=self.RotateNet(xf,x1f)
						pred2=self.RotateNet(xf,x2f)
						pred3=self.RotateNet(xf,x3f)
						pred4=self.RotateNet(x1f,x1f)
						pred5=self.RotateNet(x1f,x2f)
						pred6=self.RotateNet(x1f,x3f)
						pred7=self.RotateNet(x2f,x2f)
						pred8=self.RotateNet(x2f,x3f)
						pred9=self.RotateNet(x3f,x3f)
						SS_loss = nn.CrossEntropyLoss()(pred0,0) + nn.CrossEntropyLoss()(pred0,1) + nn.CrossEntropyLoss()(pred0,2) + nn.CrossEntropyLoss()(pred0,3) + \
								nn.CrossEntropyLoss()(pred0,4) + nn.CrossEntropyLoss()(pred0,5) + nn.CrossEntropyLoss()(pred0,6) + \
								nn.CrossEntropyLoss()(pred0,7) + nn.CrossEntropyLoss()(pred0,8) + nn.CrossEntropyLoss()(pred0,9)
						loss_dict['SS_Dloss']=float(SS_loss)
						self.optim_SS.step()"""

					# if(batch_idx+1>=N_CRITIC):
					# Logging related
					if batch_idx % self.opts.image_interval == 0 or (
							batch_idx < 1000 and batch_idx % 25 == 0):
						
						if(not os.path.exists(f'./EXP-FINAL/test-train/logs/images/test-train/{batch_idx}/')):
							os.mkdir(f'./EXP-FINAL/test-train/logs/images/test-train/{batch_idx}/')

						if(batch_idx==0):
							for i in range(x.size(0)):							
								torchvision.utils.save_image(x[i].unsqueeze(0),f'./EXP-FINAL/test-train/logs/images/test-train-x/face_{batch_idx0}_{i}.png',normalize=True)

						for i in range(y_hat.size(0)):
							torchvision.utils.save_image(y_hat[i].unsqueeze(0),f'./EXP-FINAL/test-train/logs/images/test-train/{batch_idx}/face_{batch_idx0}_{i}.png',normalize=True)

						if(self.opts.DA and (self.opts.oneshot!=0) and (self.opts.GfromZ!=0)):
							self.parse_and_log_images_MULTI(id_logs, x, GT, y_hat, title='images/train/faces')
						elif(self.opts.Encv2!=0):
							with torch.no_grad():
								z=torch.randn(x.size(0),512).cuda()
								z=self.Znet(z)
								y_random,_ = self.net.forward(x, return_latents=True,latentz=z)
								self.parse_and_log_images(id_logs, x, y_random, y_hat, title='images/train/faces')
						elif(self.opts.fakeY):
							# print('11',fake_real.size())
							self.parse_and_log_images(id_logs, x, fake_real, y_hat, title='images/train/faces')
						else:
							self.parse_and_log_images(id_logs, x, GT, y_hat, title='images/train/faces')
					if self.global_step % self.opts.board_interval == 0:
						self.print_metrics(loss_dict, prefix='train')
						self.log_metrics(loss_dict, prefix='train')

					# Validation related
					val_loss_dict = None
					if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
						val_loss_dict = self.validate()
						if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
							self.best_val_loss = val_loss_dict['loss']
							self.checkpoint_me(val_loss_dict, is_best=True)

					if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
						if val_loss_dict is not None:
							self.checkpoint_me(val_loss_dict, is_best=False)
						else:
							self.checkpoint_me(loss_dict, is_best=False)

					if self.global_step == self.opts.max_steps:
						print('OMG, finished training!')
						break
					self.global_step += 1
				# self.parse_and_log_images(id_logs, x, GT, y_hat, title=f'images/test-train/faces_{batch_idx}')
				# for i in range(y_hat.size(0)):
				# 	torchvision.utils.save_image(y_hat[i].unsqueeze(0),f'./EXP-FINAL/test-train/logs/images/test-train/face_{batch_idx0}_{i}.png',normalize=True)


	def SS_train_only(self):
		self.RotateNet.train()
		N_CRITIC=1
		while self.global_step < self.opts.max_steps:
			rot_labels = torch.zeros(4*self.opts.batch_size).cuda()
			for i in range(4*self.opts.batch_size):
				if i < self.opts.batch_size:
					rot_labels[i] = 0
				elif i < 2*self.opts.batch_size:
					rot_labels[i] = 1
				elif i < 3*self.opts.batch_size:
					rot_labels[i] = 2
				else:
					rot_labels[i] = 3
			rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
			
			for batch_idx, batch in enumerate(self.train_dataloader):
				x, GT, x1, x2, x3 = batch
				x, GT, x1, x2, x3 = x.to(self.device).float(), GT.to(self.device).float(), x1.to(self.device).float(), x2.to(self.device).float(), x3.to(self.device).float()
				y=x
				
				# self.parse_and_log_images(None, x1, x2, x3, title='images/')
				# raise RuntimeError
				# if(self.opts.SS!=0):
				requires_grad(self.net, False)
				# requires_grad(self.Discriminator, False)
				requires_grad(self.RotateNet,True)
				self.optim_SS.zero_grad()
				_,_,xf=self.net.encoder.intermediate_encode(x)
				_,_,x1f=self.net.encoder.intermediate_encode(x1)
				_,_,x2f=self.net.encoder.intermediate_encode(x2)
				_,_,x3f=self.net.encoder.intermediate_encode(x3)
				# print('5')
				pred0=self.RotateNet(xf)#0
				pred1=self.RotateNet(x1f)#1
				pred2=self.RotateNet(x2f)#2
				pred3=self.RotateNet(x3f)#3
				source = torch.cat((pred0,pred1,pred2,pred3),dim=0)
				# print('6',source.size())
				
				CEloss=F.binary_cross_entropy_with_logits
				
				# print(rot_labels.size())
				SS_loss = torch.sum(CEloss(input = source,
                                    target = rot_labels))
				# print('7')
				# loss_dict['SS_Dloss']=float(SS_loss)
				if (self.global_step % 10==0):
					print(f'iter:{self.global_step} | ss loss:{float(SS_loss):.3f}')
				SS_loss.backward()
				self.optim_SS.step()


				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					save_name = 'iteration_SS_{}.pt'.format(self.global_step)
					save_dict = self.__get_save_dict_SS()
					checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
					torch.save(save_dict, checkpoint_path)
					
				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break
				self.global_step += 1

	def load_D_weights(self):
		print('Loading D weights from pretrained!')
		
		if(self.opts.D_path is not None):
			ckpt = torch.load(self.opts.D_path, map_location='cpu')
			# print(ckpt['state_dict'])
			self.Discriminator.load_state_dict(ckpt['state_dict'], strict=True)
			print('D load from pretrained model')
		else:
			ckpt = torch.load(self.opts.stylegan_weights)
			print(ckpt.keys())
			self.Discriminator.load_state_dict(ckpt['d'], strict=False)
			print('D load from pretrained styleGAN2')
	def load_SS_weights(self):
		print('Loading SS weights from pretrained!')
		
		# if(self.opts.SS_path is not None):
		ckpt = torch.load('/data1/yuhe.ding/Code/psp/exp/exp-35-SS-orig/checkpoints/iteration_SS_2000.pt', map_location='cpu')
		# print(ckpt['state_dict'])
		self.RotateNet.load_state_dict(ckpt['state_dict'], strict=True)
		print('SS load from pretrained model iteration_SS_3200.pt')
		# else:
		# 	ckpt = torch.load(self.opts.stylegan_weights)
		# 	print(ckpt.keys())
		# 	self.Discriminator.load_state_dict(ckpt['d'], strict=False)
		# 	print('D load from pretrained styleGAN2')
	def load_Z_weights(self):
		
		if(self.opts.Zpath is not None):
			print('Loading Z weights from pretrained!')
			ckpt = torch.load(self.opts.Zpath, map_location='cpu')
			# print(ckpt['state_dict'])
			self.Znet.load_state_dict(ckpt['state_dict'], strict=True)
			print('Z load from pretrained model ')	


	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if(self.opts.SS!=0):
				x, GT, x1, x2, x3 = batch
				x, GT, x1, x2, x3 = x.to(self.device).float(), GT.to(self.device).float(), x1.to(self.device).float(), x2.to(self.device).float(), x3.to(self.device).float()
				y=x
			else:
				x, GT = batch
				x, GT = x.to(self.device).float(), GT.to(self.device).float()
				y=x

			with torch.no_grad():
				x, GT = x.to(self.device).float(), GT.to(self.device).float()
				y=x
				y_hat, latent = self.net.forward(x, return_latents=True)
				if(self.opts.SS!=0):
					loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, GT,val=True,x1=x1,x2=x2,x3=x3)
				else:
					loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, GT,val=True)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, GT, y_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# os.makedirs(os.path.join(self.opts.exp_dir,'inference_results'))
			# im_save_path = os.path.join(os.path.join(self.opts.exp_dir,'inference_results'), os.path.basename(str(batch_idx)))
			# from PIL import Image
			# from utils.common import tensor2im
			# Image.fromarray(np.array(tensor2im(y_hat[0]))).save(im_save_path+'.png')
			# print('1')
			# raise RuntimeError

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
		if(self.opts.GAN_lambda!=0):
			save_name = 'best_D_model.pt' if is_best else 'iteration_D_{}.pt'.format(self.global_step)
			save_dict = self.__get_save_dict_D()
			checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
			torch.save(save_dict, checkpoint_path)
			with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
				if is_best:
					f.write('**Best D**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
				else:
					f.write('D Step - {}, \n{}\n'.format(self.global_step, loss_dict))
		if(self.opts.SS!=0):
			save_name = 'best_SS_model.pt' if is_best else 'iteration_SS_{}.pt'.format(self.global_step)
			save_dict = self.__get_save_dict_SS()
			checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
			torch.save(save_dict, checkpoint_path)
			with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
				if is_best:
					f.write('**Best SS**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
				else:
					f.write('SS Step - {}, \n{}\n'.format(self.global_step, loss_dict))
		if((self.opts.remapping!=0) or (self.opts.Encv2!=0)):
			save_name = 'best_Z_model.pt' if is_best else 'iteration_Z_{}.pt'.format(self.global_step)
			save_dict = self.__get_save_dict_Z()
			checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
			torch.save(save_dict, checkpoint_path)
		if(self.opts.StyleEnc!=0):
			save_name = 'best_SE_model.pt' if is_best else 'iteration_SE_{}.pt'.format(self.global_step)
			save_dict = self.__get_save_dict_SE()
			checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
			torch.save(save_dict, checkpoint_path)

		
	def configure_optimizers(self):
		if(self.opts.Zonly!=0):
			params = list(self.Znet.parameters())
		else:
			params = list(self.net.encoder.parameters())
			if(self.opts.withNCE):
				params = list(itertools.chain(self.net.encoder.parameters(),self.netF.parameters()))
			if self.opts.train_decoder:
				params += list(self.net.decoder.parameters())
			if((self.opts.remapping!=0) or (self.opts.Encv2!=0)):
				params+=list(self.Znet.parameters())
			if(self.opts.StyleEnc!=0 and (not self.opts.DA)):
				params+=list(self.StyleNet.parameters())
		# if(self.opts.MI!=0):
		# 	params+=list(self.StatisticNet_X2Z.parameters())
		# 	params+=list(self.StatisticNet_Z2Y.parameters())
		
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer
	def configure_optim_D(self):
		params = list(self.Discriminator.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer
	def configure_optim_SS(self):
		params = list(self.RotateNet.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		if(self.opts.DA):
			dataset_args = data_configs.DATASETS['DA']

		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		if(self.opts.data):
			train_dataset = ImagesDataset(source_root=self.opts.source,
									  target_root=self.opts.target,
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
			# test_dataset = ImagesDataset(source_root=self.opts.source,
			# 							target_root=self.opts.target,
			# 							source_transform=transforms_dict['transform_source'],
			# 							target_transform=transforms_dict['transform_test'],
			# 							opts=self.opts)
			test_dataset = ImagesDataset(source_root='../../DATA/content',
										target_root='../../DATA/content',
										source_transform=transforms_dict['transform_source'],
										target_transform=transforms_dict['transform_test'],
										opts=self.opts)
			print(f'datasets loaded from {self.opts.source}')
		else:
			train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
										target_root=dataset_args['train_target_root'],
										source_transform=transforms_dict['transform_source'],
										target_transform=transforms_dict['transform_gt_train'],
										opts=self.opts)
			test_dataset = ImagesDataset(source_root='../../DATA/content',
										target_root='../../DATA/content',
										source_transform=transforms_dict['transform_source'],
										target_transform=transforms_dict['transform_test'],
										opts=self.opts)
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calculate_NCE_loss(self, src, tgt):
		n_layers=3
		# print(n_layers,self.nce_layers)
		##获取tgt图像的中间特征
		feat_q = self.net.encoder.intermediate_encode(tgt)
		# print(len(feat_q))
		# for i in range(len(feat_q)):
		#     print(feat_q[i].size())

		feat_k = self.net.encoder.intermediate_encode(src)
		feat_k_pool, sample_ids = self.netF(feat_k, num_patches=256)
		feat_q_pool, _ = self.netF(feat_q, 256, sample_ids)
		# print(len(feat_k_pool))
		# for i in range(len(feat_k_pool)):
		#     print(feat_k_pool[i].size())
		# for i in range(len(sample_ids)):
		#     print(sample_ids[i].size())
		# print(sample_ids.size())
		total_nce_loss = 0.0
		for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
			# print(f_q.size(),f_k.size())
			loss = crit(f_q, f_k) 
			total_nce_loss += loss.mean()
			# raise RuntimeError

		return total_nce_loss / n_layers

	def calc_loss(self, x, y, y_hat, latent,GT=None,val=False,x1=None,x2=None,x3=None):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			if(self.opts.GrayL2!=0):
				y1=y.mean(dim=1)
				y2 = y_hat.mean(dim=1)
				# print(y1.size(),y2.size())
				loss_l2 = F.mse_loss(y1,y2)
			else:
				loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda
		if (self.opts.withNCE):
			loss_NCE=self.calculate_NCE_loss(y_hat,x)
			loss+=loss_NCE*self.opts.nce_lambda
			loss_dict['patchNCE'] = float(loss_NCE)	
		if (self.opts.KD):
			loss_channel,loss_batch,loss_pixel,loss_KD = self.da_loss(y_hat,x)
			loss_dict['loss_channel'] = float(loss_channel)
			loss_dict['loss_batch'] = float(loss_batch)
			loss_dict['loss_pixel'] = float(loss_pixel)
			loss_dict['loss_KD'] = float(loss_KD)
			loss+= loss_channel*self.opts.lc_lambda
			loss+= loss_batch*self.opts.lb_lambda
			loss+= loss_pixel*self.opts.lp_lambda
			loss+= loss_KD*self.opts.KD_lambda	
		if (self.opts.structure_loss > 0):
			# struc_loss=0
			for layer in range(self.opts.structure_loss):
				# _, latent_med_sor = generator_source(noise, swap=True, swap_layer_num=layer+1)
				structure_x = self.net.forward(x, struc=True, structure_layer_num=layer+1)
				structure_y_hat = self.net.forward(y_hat, struc=True, structure_layer_num=layer+1)
				if(layer==0):
					struc_loss = F.mse_loss(structure_x, structure_y_hat)*self.opts.struc_lambda
				else:
					struc_loss += F.mse_loss(structure_x, structure_y_hat)*self.opts.struc_lambda
			loss+=struc_loss
			loss_dict['struc_loss'] = float(struc_loss)
		if((self.opts.GAN_lambda!=0) or (self.opts.D_lambda!=0)):
			out = self.Discriminator(y_hat)
			loss_G = F.softplus(-out).mean()
			if(self.opts.GAN_lambda!=0):
				loss+=loss_G*self.opts.GAN_lambda
			else:
				loss+=loss_G*self.opts.D_lambda
			loss_dict['loss_G'] = float(loss_G)
		if (self.opts.Rel_lambda):
			sfm=nn.Softmax()
			sim = nn.CosineSimilarity()
			kl_loss = nn.KLDivLoss()
			# distance consistency loss
			# with torch.set_grad_enabled(False):
			# z = torch.randn(args.feat_const_batch, args.latent, device=device)
			size=x.size()[0]
			feat_ind = numpy.random.randint(1, self.net.decoder.n_latent - 1, size=size)
			# print(feat_ind)

			# computing source distances
			source_sample, feat_source = self.net.forward(x, return_feats=True)
			# print(len(feat_source),feat_source[0].size())
			dist_source = torch.zeros(
				[size, size - 1]).cuda()

			# iterating over different elements in the batch
			for pair1 in range(size):
				tmpc = 0
				# comparing the possible pairs
				for pair2 in range(size):
					if pair1 != pair2:
						# print(feat_ind[pair1],pair1,feat_ind[pair1],pair2)
						anchor_feat = torch.unsqueeze(
							feat_source[feat_ind[pair1]][pair1].reshape(-1), 0)
						compare_feat = torch.unsqueeze(
							feat_source[feat_ind[pair1]][pair2].reshape(-1), 0)
						dist_source[pair1, tmpc] = sim(
							anchor_feat, compare_feat)
						tmpc += 1
			dist_source = sfm(dist_source)

			# computing distances among target generations
			_, feat_target = self.net.forward(y_hat, return_feats=True)
			dist_target = torch.zeros(
				[size, size - 1]).cuda()

			# iterating over different elements in the batch
			for pair1 in range(size):
				tmpc = 0
				for pair2 in range(size):  # comparing the possible pairs
					if pair1 != pair2:
						anchor_feat = torch.unsqueeze(
							feat_target[feat_ind[pair1]][pair1].reshape(-1), 0)
						compare_feat = torch.unsqueeze(
							feat_target[feat_ind[pair1]][pair2].reshape(-1), 0)
						dist_target[pair1, tmpc] = sim(anchor_feat, compare_feat)
						tmpc += 1
			dist_target = sfm(dist_target)
			rel_loss = self.opts.Rel_lambda * \
				kl_loss(torch.log(dist_target), dist_source) # distance consistency loss 
			loss+=rel_loss
			loss_dict['rel_loss'] = float(rel_loss)
		if (self.opts.TexPr_lambda!=0):
			with torch.set_grad_enabled(False):
				fake_Real,fake_Real_feats = self.Teacher.forward(x,return_feats=True)
			_, feat = self.net.forward(x, return_feats=True)
			for i in range(3,len(fake_Real_feats)):
				if(i==3):
					TexPr_feat_Loss = F.mse_loss(fake_Real_feats[i],feat[i])
				else:
					TexPr_feat_Loss += F.mse_loss(fake_Real_feats[i],feat[i])
			loss+=TexPr_feat_Loss*self.opts.TexPr_lambda
			loss_dict['TexPr_feat_Loss'] = float(TexPr_feat_Loss)
		if(self.opts.MI!=0):
			if(not val):
				xc1,xc2,xc3 = self.net.encoder.intermediate_encode(x)
				yc1,yc2,yc3 = self.net.encoder.intermediate_encode(GT)
				self.mine_x2c1.update(x,xc1,x,yc1)

				xc1,xc2,xc3 = self.net.encoder.intermediate_encode(x)
				yc1,yc2,yc3 = self.net.encoder.intermediate_encode(GT)
				self.mine_x2c3.update(x,xc3,x,yc3)

			xc1,xc2,xc3 = self.net.encoder.intermediate_encode(x)
			yc1,yc2,yc3 = self.net.encoder.intermediate_encode(GT)
			E_pos, E_neg,_,_ = self.mine_x2c1.score(x,xc1,x,yc1)
			I_xc1 = E_pos-E_neg
			E_pos_y, E_neg_y,_,_ = self.mine_x2c3.score(x,xc3,x,yc3)
			I_xc3 = E_pos_y-E_neg_y

			loss+=(I_xc1-I_xc3)*self.opts.MI
			loss_dict['IB'] = float(I_xc1-I_xc3)
		if(self.opts.fliploss!=0):
			x_flip = self.Flip(x)
			y_f_hat=self.net.forward(x_flip)
			y_hat_flip = self.Flip(y_hat)
			flip_loss = F.mse_loss(y_f_hat,y_hat_flip)
			loss+=flip_loss*self.opts.fliploss
			loss_dict['flip_loss'] = float(flip_loss)
			# self.parse_and_log_images(id_logs,y_hat,y_hat_flip,y_f_hat,'./')

			# raise RuntimeError
		if(self.opts.SS!=0):
			self.RotateNet.eval()
			# requires_grad(self.RotateNet,False)
			rot_labels = torch.zeros(4*x.size()[0]).cuda()
			for i in range(4*self.opts.batch_size):
				if i < self.opts.batch_size:
					rot_labels[i] = 0
				elif i < 2*self.opts.batch_size:
					rot_labels[i] = 1
				elif i < 3*self.opts.batch_size:
					rot_labels[i] = 2
				else:
					rot_labels[i] = 3
			rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
			
			_,_,xf=self.net.encoder.intermediate_encode(x)
			_,_,x1f=self.net.encoder.intermediate_encode(x1)
			_,_,x2f=self.net.encoder.intermediate_encode(x2)
			_,_,x3f=self.net.encoder.intermediate_encode(x3)
			# print('5')
			pred0=self.RotateNet(xf)#0
			pred1=self.RotateNet(x1f)#1
			pred2=self.RotateNet(x2f)#2
			pred3=self.RotateNet(x3f)#3
			source = torch.cat((pred0,pred1,pred2,pred3),dim=0)
			CEloss=F.binary_cross_entropy_with_logits
			SS_loss = torch.sum(CEloss(input = source, target = rot_labels))
			loss+=SS_loss*self.opts.SS
			loss_dict['SS'] = float(SS_loss)
		if(self.opts.swap_lambda!=0):
			x2y_swap = self.net.swap(x,GT,12)
			swap_loss = F.mse_loss(x2y_swap,y_hat)
			x2y2x_swap = self.net.swap(y_hat,x,12)
			swap_cyc_loss = F.mse_loss(x2y2x_swap,x)
			loss+=self.opts.swap_lambda*(swap_loss+swap_cyc_loss)
			loss_dict['swap_loss'] = float(self.opts.swap_lambda*(swap_loss+swap_cyc_loss))
			# self.parse_and_log_images(None,swap,x,GT,title='images')
			# raise RuntimeError
		if(self.opts.StyleEnc!=0):
			s_x_enc = self.StyleNet(x).view(x.size(0),-1).unsqueeze(1).repeat(1, 6, 1)
			s_GT_enc = self.StyleNet(GT).view(x.size(0),-1).unsqueeze(1).repeat(1, 6, 1)

			s_x = torch.randn(s_x_enc.size()).float().cuda()
			s_GT = torch.randn(s_GT_enc.size()).float().cuda()
			#whitin domain
			x_rec, _ = self.net.forward(x, return_latents=True,latentz=s_x_enc)
			x_rec_loss = F.mse_loss(x,x_rec)

			GT_rec, _ = self.net.forward(GT, return_latents=True,latentz=s_GT_enc)
			GT_rec_loss = F.mse_loss(GT,GT_rec)

			#cross domain
			x2GT, _ = self.net.forward(x, return_latents=True,latentz=s_GT+torch.randn(s_GT.size()).float().cuda())
			GT2x, _ = self.net.forward(GT, return_latents=True,latentz=s_x+torch.randn(s_x.size()).float().cuda())

			#cyc encode recon
			s_x2GT = self.StyleNet(x2GT).view(x.size(0),-1).unsqueeze(1).repeat(1, 6, 1)
			s_GT2x = self.StyleNet(GT2x).view(x.size(0),-1).unsqueeze(1).repeat(1, 6, 1)
			s_x_rec_loss = F.mse_loss(s_x,s_GT2x)
			s_GT_rec_loss = F.mse_loss(s_GT,s_x2GT)
			styLoss = x_rec_loss + GT_rec_loss + s_x_rec_loss + s_GT_rec_loss
			loss+=styLoss*self.opts.StyleEnc
			# print(s_x.size(),s_GT.size())


			loss_dict['style loss'] = float(styLoss)
			if(self.opts.KL_lambda!=0):
				def compute_kl(mu):
					mu2=torch.pow(mu,2)
					return torch.mean(mu2)
				KLLoss= compute_kl(s_x) + compute_kl(s_GT) + compute_kl(s_x2GT)+ compute_kl(s_GT2x)
				
				loss+=KLLoss * self.opts.KL_lambda
				loss_dict['KL Loss']=float(KLLoss)

			if(self.opts.DisConsis!=0):
				# sfm = nn.Softmax(dim=1)
				# kl_loss = nn.KLDivLoss()
				sim = nn.CosineSimilarity()

				x_rec, feat_x = self.net.forward(x, return_feats=True,latentz=s_x)
				GT_rec, feat_GT = self.net.forward(GT, return_feats=True,latentz=s_GT)
				x2GT, feat_x2GT = self.net.forward(x, return_feats=True,latentz=s_GT)
				GT2x, feat_GT2x = self.net.forward(GT, return_feats=True,latentz=s_x)

				target_dis=sim(feat_x[0].view(feat_x[0].size(0),-1), feat_GT[0].view(feat_x[0].size(0),-1))
				source_dis=sim(feat_x2GT[0].view(feat_x2GT[0].size(0),-1), feat_GT2x[0].view(feat_GT2x[0].size(0),-1))

				for i in range(len(feat_x)-1):
					target_dis += sim(feat_x[i+1].view(feat_x[i+1].size(0),-1),feat_GT[i+1].view(feat_x[i+1].size(0),-1))
					source_dis += sim(feat_x2GT[i+1].view(feat_x2GT[i+1].size(0),-1),feat_GT2x[i+1].view(feat_GT2x[i+1].size(0),-1))
				distance_consis_loss = nn.L1Loss()(source_dis,target_dis)
				loss += self.opts.DisConsis*(distance_consis_loss)

				loss_dict['distance_consistancy loss'] = float(distance_consis_loss)
		if(self.opts.GfromZ!=0):
			if (self.opts.DisConsis!=0):
				sim = nn.CosineSimilarity()
				sfm = nn.Softmax(dim=1)
				kl_loss = nn.KLDivLoss()
				img_ind = random.randint(0,self.opts.batch_size-1)
				z1=torch.randn(3,6,512).float().cuda()
				y1_hat, feat_z1 = self.net.forward(x[img_ind].unsqueeze(0).repeat(3,1,1,1), return_feats=True,latentz=z1)
				dis_source = torch.zeros([3,2]).cuda()
				for pair1 in range(3):
					tmpc=0
					anchor_feat = z1[pair1].view(-1).unsqueeze(0)
					for pair2 in range(3):
						if(pair1!=pair2):
							compare_feat = z1[pair2].view(-1).unsqueeze(0)
							dis_source[pair1,tmpc] = sim(anchor_feat,compare_feat)
							# raise RuntimeError
							tmpc+=1
				dis_source = sfm(dis_source)
				# print(dis_source.size())
				
				dis_target = torch.zeros([3,2]).cuda()
				feat_ind=random.randint(18-self.opts.Zlevel-1,len(feat_z1)-1)

				for p1 in range(3):
					tmpc = 0
					anchor_feat = feat_z1[feat_ind][p1].view(-1).unsqueeze(0)
					for p2 in range(3):
						if(p1!=p2):
							compare_feat = feat_z1[feat_ind][p2].view(-1).unsqueeze(0)
							dis_target[p1,tmpc] = sim(anchor_feat, compare_feat)
							# print(len(feat_z1))
							tmpc+=1
				
				dis_target = sfm(dis_target)
				disloss = kl_loss(torch.log(dis_target),dis_source)

				loss+=self.opts.DisConsis * disloss
				loss_dict['distance_consistancy loss'] = float(disloss*self.opts.DisConsis)
			if(self.opts.MaxDist!=0):	
				sim = nn.CosineSimilarity()			
				z1=torch.randn(2,self.opts.Zlevel,512).float().cuda()
				img_ind = random.randint(0,self.opts.batch_size-1)
				y1_hat, feat_z1 = self.net.forward(x[img_ind].unsqueeze(0).repeat(2,1,1,1), return_feats=True,latentz=z1)
				# dis_target = torch.zeros([3,2]).cuda()
				feat_ind=random.randint(18-self.opts.Zlevel-1,len(feat_z1)-1)
				anchor_feat = feat_z1[feat_ind][0].view(-1).unsqueeze(0)
				compare_feat = feat_z1[feat_ind][1].view(-1).unsqueeze(0)
				dist = -sim(anchor_feat, compare_feat).mean()
				
				# dis_target = sfm(dis_target)
				# print(dist.size())
				loss+=self.opts.MaxDist * dist
				loss_dict['distance_consistancy loss'] = float(dist*self.opts.MaxDist)
		if((self.opts.remapping!=0) and (self.opts.DisConsis!=0)):
			sim = nn.CosineSimilarity()
			sfm = nn.Softmax(dim=1)
			kl_loss = nn.KLDivLoss()
			img_ind = random.randint(0,self.opts.batch_size-1)
			z1=torch.randn(3,512).float().cuda()
			# print(z1.size())
			z1=self.Znet(z1)
			# print(z1.size())
			z1=z1.unsqueeze(1).repeat(1, self.opts.Zlevel, 1)
			# print(z1.size())
			y1_hat, feat_z1 = self.net.forward(x[img_ind].unsqueeze(0).repeat(3,1,1,1), return_feats=True,latentz=z1)
			# print(y1_hat.size())
			dis_source = torch.zeros([3,2]).cuda()
			for pair1 in range(3):
				tmpc=0
				anchor_feat = z1[pair1].view(-1).unsqueeze(0)
				for pair2 in range(3):
					if(pair1!=pair2):
						compare_feat = z1[pair2].view(-1).unsqueeze(0)
						dis_source[pair1,tmpc] = sim(anchor_feat,compare_feat)
						# raise RuntimeError
						tmpc+=1
			dis_source = sfm(dis_source)
			# print(dis_source.size())
			
			dis_target = torch.zeros([3,2]).cuda()
			feat_ind=random.randint(18-self.opts.Zlevel-1,len(feat_z1)-1)

			for p1 in range(3):
				tmpc = 0
				anchor_feat = feat_z1[feat_ind][p1].view(-1).unsqueeze(0)
				for p2 in range(3):
					if(p1!=p2):
						compare_feat = feat_z1[feat_ind][p2].view(-1).unsqueeze(0)
						dis_target[p1,tmpc] = sim(anchor_feat, compare_feat)
						# print(len(feat_z1))
						tmpc+=1
			
			dis_target = sfm(dis_target)
			disloss = kl_loss(torch.log(dis_target),dis_source)

			loss+=self.opts.DisConsis * disloss
			loss_dict['distance_consistancy loss'] = float(disloss*self.opts.DisConsis)
		if(self.opts.Encv2!=0):
			if(self.opts.Encv3!=0):
				x2y_swap = self.net.swap(x,GT,9)
				self.parse_and_log_images(None, x, GT, x2y_swap, title='images/')
				raise RuntimeError
			
			z=torch.randn(x.size(0),512).float().cuda()
			z=self.Znet(z)
			yz_hat, _ = self.net.forward(x, return_feats=True,latentz=z)
			yz_out = self.Discriminator(yz_hat)
			loss_yz_G = F.softplus(-yz_out).mean()
			loss+=self.opts.GAN_lambda * loss_yz_G
			loss_dict['yz_G_loss'] = float(loss_yz_G)
			if(self.opts.mappingloss!=0):
				code = self.net.encode(x)
				rec_code = self.net.encode(y_hat)
				
				# raise RuntimeError('11')
				if(self.opts.coderec!=0):
					code_orig = torch.cat((code[:,:18-self.opts.Zlevel,:],z),dim=1)
					z_rec_loss = F.mse_loss(code_orig, rec_code)
				else:
					z_rec_loss = F.mse_loss(rec_code[:,18-self.opts.Zlevel:,:],z)
				loss+=self.opts.Encv2*z_rec_loss
				loss_dict['z_rec_loss'] = float(z_rec_loss)
				

			if(self.opts.DisConsis!=0):
				sim = nn.CosineSimilarity()
				sfm = nn.Softmax(dim=1)
				kl_loss = nn.KLDivLoss()
				# ind = random.randint(0,self.opts.Zlevel-1)
				# z=torch.randn(x.size(0),512).float().cuda()
				# z=self.Znet(z)
				# print(z.size())
				# code = self.net.encode(x)
				codes=code[:,18-self.opts.Zlevel:,:]
				# print(codes.size())
				if(x.size(0)>2):
					dis_source = torch.zeros([x.size(0),x.size(0)-1]).cuda()
					for pair1 in range(x.size(0)):
						tmpc=0
						anchor_feat = z[pair1].view(-1).unsqueeze(0)
						# print(anchor_feat.size())
						for pair2 in range(x.size(0)):
							if(pair1!=pair2):
								compare_feat = z[pair2].view(-1).unsqueeze(0)
								dis_source[pair1,tmpc] = sim(anchor_feat,compare_feat)
								# raise RuntimeError
								tmpc+=1
					dis_source = sfm(dis_source)
					# print(dis_source.size())
					
					dis_target = torch.zeros([x.size(0),x.size(0)-1]).cuda()
					# feat_ind=random.randint(18-self.opts.Zlevel-1,len(feat_z1)-1)
					for p1 in range(x.size(0)):
						tmpc = 0
						anchor_feat = codes[p1].view(-1).unsqueeze(0)
						for p2 in range(x.size(0)):
							if(p1!=p2):
								compare_feat =codes[p2].view(-1).unsqueeze(0)
								dis_target[p1,tmpc] = sim(anchor_feat, compare_feat)
								# print(len(feat_z1))
								tmpc+=1
					dis_target = sfm(dis_target)
					disloss = kl_loss(torch.log(dis_target),dis_source)

					loss+=self.opts.DisConsis * disloss
					loss_dict['disloss'] = float(disloss)
				elif(x.size(0)==2):
					dis_z = sim(z[0].view(-1).unsqueeze(0), z[1].view(-1).unsqueeze(0))
					dis_code = sim(codes[0].view(-1).unsqueeze(0), codes[1].view(-1).unsqueeze(0))
					disloss = F.mse_loss(dis_z,dis_code)

					loss+=self.opts.DisConsis * disloss
					loss_dict['disloss'] = float(disloss)

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs
	def Flip(self,input):
		bs,c,w,h=input.size()
		out = torch.tensor((), dtype=torch.float32).new_ones(input.size()).cuda()
		for i in range(w):
			out[:,:,:,i]=input[:,:,:,w-i-1]
		return out
	def RandomCover(self,x,size=64):
		w,h = x.size()[2], x.size()[3]
		m = random.randint(0,w-size-1)
		n = random.randint(0,h-size-1)
		# x_cover=torch.zeros(x.size()).float().cuda()
		x_cover=x.clone()
		for i in range(size):
			for j in range(size):
				x_cover[:,:,m+i,n+j]=0.
		return x_cover

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		display_count=x.size()[0]
		for i in range(display_count):
			# print(GT[i].min(),GT[i].max())
			cur_im_data = {
				'input_face': common.log_input_image(x[i],self.opts),
				'target_face': common.log_input_image(y[i],self.opts),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)
	def parse_and_log_images_MULTI(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		with torch.no_grad():
			z1=torch.randn(x.size()[0],6,512).float().cuda()
			y1, _ = self.net.forward(x, return_latents=True,latentz=z1)
			z2=torch.randn(x.size()[0],6,512).float().cuda()
			y2, _ = self.net.forward(x, return_latents=True,latentz=z2)
			z3=torch.randn(x.size()[0],6,512).float().cuda()
			y3, _ = self.net.forward(x, return_latents=True,latentz=z3)
			z4=torch.randn(x.size()[0],6,512).float().cuda()
			y4, _ = self.net.forward(x, return_latents=True,latentz=z4)
			z5=torch.randn(x.size()[0],6,512).float().cuda()
			y5, _ = self.net.forward(x, return_latents=True,latentz=z5)
		im_data = []
		display_count=x.size()[0]
		for i in range(display_count):
			# print(GT[i].min(),GT[i].max())
			cur_im_data = {
				'input_face': common.log_input_image(x[i],self.opts),
				'y1': common.tensor2im(y1[i]),
				'y2': common.tensor2im(y2[i]),
				'y3': common.tensor2im(y3[i]),
				'y4': common.tensor2im(y4[i]),
				'y5': common.tensor2im(y5[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images_MULTI(title, im_data=im_data, subscript=subscript)
	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def log_images_MULTI(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces_multi(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
	def __get_save_dict_D(self):
		save_dict = {
			'state_dict': self.Discriminator.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict
	def __get_save_dict_SS(self):
		save_dict = {
			'state_dict': self.RotateNet.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict
	def __get_save_dict_Z(self):
		save_dict = {
			'state_dict': self.Znet.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict
	def __get_save_dict_SE(self):
		save_dict = {
			'state_dict': self.StyleNet.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict
