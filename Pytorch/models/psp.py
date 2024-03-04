"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')
import math

import torch
from torch import nn
from torch.nn import init
from models.encoders import psp_encoders
from models.stylegan2.model import Generator, Discriminator
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt
class Normalize(nn.Module):
	def __init__(self, power=2):
		super(Normalize, self).__init__()
		self.power = power

	def forward(self, x):
		norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
		out = x.div(norm + 1e-7)
		return out
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net
def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            # print(input_nc)
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            # print(feat_reshape.size())
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    #randperm:返回一个0~n-1的数组，第一个参数为上界
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    # print(patch_id.size())
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    # print(patch_id.size())
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                # print(x_sample.size())
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            # raise RuntimeError
        return return_feats, return_ids


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			# print(get_keys(ckpt, 'encoder'))
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
			
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)
			
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			# print(ckpt.keys())
			# raise RuntimeError
			if(self.opts.stylegan_load):
				self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
			if self.opts.learn_in_w:
				self.__load_latent_avg(ckpt, repeat=1)
			else:
				self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None, struc=False,structure_layer_num=1,
				return_feats=False, statis=False, latentz=None,layer=6):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# print(latentz)
			if(latentz is not None):
				# print(codes.size(),latentz.size())
				layer = codes.size(1) - latentz.size(1)
				if(self.opts.CrossCat!=0):
					for i in range(6):
						codes[:,i*2+1,:] = latentz[:,i,:]
					# print(codes.size())
				else:
					codes=torch.cat((codes[:,:layer,:],latentz),dim=1)
				# raise RuntimeError
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		# print(codes.size())
		# raise RuntimeError
		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		if(return_feats):
			images, feats = self.decoder([codes],
											input_is_latent=input_is_latent,
											randomize_noise=randomize_noise,
											return_latents=return_latents,
											return_feats=return_feats)
			# print(images.size())
			return images, feats
		if(struc):
			structure_layer = self.decoder([codes],
										input_is_latent=input_is_latent,
										randomize_noise=randomize_noise,
										return_latents=return_latents,
										struc=True,
										structure_layer_num=structure_layer_num)
		elif(statis):
			feat, gram, entropy = self.decoder.calc_statistic([codes],
										input_is_latent=input_is_latent,
										randomize_noise=randomize_noise,
										return_latents=return_latents,
										struc=True,
										structure_layer_num=structure_layer_num)
		else:
			images, result_latent = self.decoder([codes],
											input_is_latent=input_is_latent,
											randomize_noise=randomize_noise,
											return_latents=return_latents)
		if(struc):
			return structure_layer
		if resize:
			images = self.face_pool(images)
		if(statis):
			return feat, gram, entropy
		if return_latents:
			return images, result_latent
		else:
			return images
	def swap(self, x, y,layer=12,resize=True, latent_mask=None, input_code=False, randomize_noise=True,
			inject_latent=None, return_latents=False, alpha=None, struc=False,structure_layer_num=1,
				return_feats=False, statis=False):
		if input_code:
			codes = x
			codesy=y
		else:
			codes = self.encoder(x)
			codesy=self.encoder(y)
			# normalize with respect to the center of an average face
			for i in range(layer,18):
				codes[:,i,:]=codesy[:,i,:]
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		# print(codes.size())
		# raise RuntimeError
		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		if(return_feats):
			images, feats = self.decoder([codes],
											input_is_latent=input_is_latent,
											randomize_noise=randomize_noise,
											return_latents=return_latents,
											return_feats=return_feats)
			return images, feats
		if(struc):
			structure_layer = self.decoder([codes],
										input_is_latent=input_is_latent,
										randomize_noise=randomize_noise,
										return_latents=return_latents,
										struc=True,
										structure_layer_num=structure_layer_num)
		elif(statis):
			feat, gram, entropy = self.decoder.calc_statistic([codes],
										input_is_latent=input_is_latent,
										randomize_noise=randomize_noise,
										return_latents=return_latents,
										struc=True,
										structure_layer_num=structure_layer_num)
		else:
			images, result_latent = self.decoder([codes],
											input_is_latent=input_is_latent,
											randomize_noise=randomize_noise,
											return_latents=return_latents)
		if(struc):
			return structure_layer
		if resize:
			images = self.face_pool(images)
		if(statis):
			return feat, gram, entropy
		if return_latents:
			return images, result_latent
		else:
			return images

	def encode(self,x):
		codes = self.encoder(x)
		# normalize with respect to the center of an average face
		if self.opts.start_from_latent_avg:
			if self.opts.learn_in_w:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		return codes

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
