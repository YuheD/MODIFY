import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import random

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from training.coach import Z_mapping
# def seed_torch(seed=1029):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
# seed_torch()
def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    if(test_opts.remapping!=0):
        Znet=Z_mapping(code_dim=test_opts.Zlevel)
        Zckpt=torch.load(test_opts.Zpath, map_location='cpu')
        print('Loading Z from checkpoint: {}'.format(test_opts.Zpath))
        Znet.load_state_dict(Zckpt['state_dict'], strict=True)
        Znet.eval()
        Znet.cuda()
        
    # if 'learn_in_w' not in opts:
    #     opts['learn_in_w'] = False
    # if 'output_size' not in opts:
    #     opts['output_size'] = 1024
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 256
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []

    if(test_opts.interpolate!=0):
        z1 = torch.randn(1,512).float().cuda()
        z2 = torch.randn(1,512).float().cuda()
        diff = (z2-z1)/test_opts.num_style

    for sty in range (test_opts.num_style):
        if(test_opts.interpolate!=0):
            z = z1+sty*diff
        else:
            z=torch.randn(1,512).float().cuda()
        # z=z.repeat(2,1)
        # print(z.size())
        global_i = 0
        for _,input_batch in (enumerate(tqdm(dataloader))):
            if global_i >= (opts.n_images):
                break
            with torch.no_grad():
                input_cuda = input_batch.cuda().float()
                tic = time.time()
                # if(test_opts.remapping!=0):
                if(test_opts.remapping):
                    if(test_opts.num_style!=1):
                        result_batch = run_on_batch(input_cuda, net, opts,Znet=Znet,z=z)
                    else:
                        result_batch = run_on_batch(input_cuda, net, opts,Znet=Znet)
                elif(test_opts.num_style!=1):
                    result_batch = run_on_batch(input_cuda, net, opts,z=z)
                else:
                    # print('1')
                    result_batch = run_on_batch(input_cuda, net, opts)
                toc = time.time()
                global_time.append(toc - tic)

            for i in range(opts.test_batch_size):
                result = tensor2im(result_batch[i])
                im_path = dataset.paths[global_i]

                # if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i],opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                        np.array(result.resize(resize_amount))], axis=1)
                                        
                # print(os.path.join(out_path_coupled, os.path.basename(im_path))[:-4]+'_'+str(sty)+'.png')
                
                if(opts.Metric==0):
                    Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path))[:-4]+'_'+str(sty)+'.png')
                
                im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
                # print(im_save_path[:-4]+'_'+str(sty)+'.png')
                # raise RuntimeError
                Image.fromarray(np.array(result)).save(im_save_path[:-4]+'_'+str(sty)+'.png')

                global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts, Znet=None,z=None):
    if(z is not None):
        if(z.size()[0]!=inputs.size()[0]):
            z=z.repeat(inputs.size()[0],1)
            # print('1',z.size())
            # raise RuntimeError
    # if opts.latent_mask is None:
    if((opts.GfromZ!=0) and (opts.num_style==1)):
        z=torch.randn(inputs.size()[0],6,512).float().cuda()
        # raise RuntimeError
    if(Znet is not None):
        if(z is None):
            z=torch.randn(inputs.size()[0],512).float().cuda()
        z=Znet(z)
        # print(z.size())
        # z=z.unsqueeze(1).repeat(1, 6, 1)
    # print(z)
    # result_batch = net.forward(inputs, randomize_noise=False, resize=opts.resize_outputs,latentz=z)
    result_batch,_ = net.forward(inputs, resize=opts.resize_outputs,latentz=z,return_latents=True)
    return result_batch

# def calc_Entropy():

if __name__ == '__main__':
    # seed_torch()
    run()
