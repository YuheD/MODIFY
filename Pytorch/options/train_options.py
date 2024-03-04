from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, default='./experiment',help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='celebs_sketch_to_face', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
		self.parser.add_argument('--sort', default=0, type=int, help='Output size of generator')
		self.parser.add_argument('--MI', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--oneshot', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--testtrain', action='store_true', help='Output size of generator')
		self.parser.add_argument('--one_path', default='../database/Photo/0/01011.png', type=str, help='Output size of generator')
		self.parser.add_argument('--Zpath', default=None, type=str, help='Output size of generator')
		self.parser.add_argument('--MultiFakeY', default=0, type=int, help='Output size of generator')
		self.parser.add_argument('--DisConsis', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--GrayL2', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--Zlevel', default=9, type=int, help='Output size of generator')
		self.parser.add_argument('--MaxDist', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--EMA', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--CrossCat', default=0, type=float, help='Output size of generator')
		self.parser.add_argument('--coderec', default=1, type=float, help='Output size of generator')

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=1, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--GAN_lambda', default=0, required=True,type=float, help='GAN loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, required=True,type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0.005, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
		self.parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
		self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--lc_lambda', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--lb_lambda', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--lp_lambda', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--KD_lambda', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--nce_lambda', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--D_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Rel_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--TexPr_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--nce_T', default=0.07, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--DA', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--withNCE', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--freezeG', default=0, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--fliploss', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--swap_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--GfromZ', default=0, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--remapping', default=0, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--StyleEnc', default=0, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--KL_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Encv2', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Encv3', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--mappingloss', default=1, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Zonly', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Gray', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		
		self.parser.add_argument('--freezeE', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--structure_loss', default=0, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--struc_lambda', default=1, type=int, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--KD', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--add_branch', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Notadd', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--fakeY', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--use_ada', default=False, type=bool, help='Whether to train the decoder model')
		
		self.parser.add_argument('--data', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--source', default='../database/Photo/0', type=str, help='Whether to train the decoder model')
		self.parser.add_argument('--target', default='../database/MetFaces/train/0', type=str, help='Whether to train the decoder model')

		self.parser.add_argument('--stylegan_load', action='store_true', help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--D_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--SS', default=0, type=float, help='Path to pSp model checkpoint')

		self.parser.add_argument('--max_steps', default=150000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=2500, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=5000, type=int, help='Model checkpoint interval')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
