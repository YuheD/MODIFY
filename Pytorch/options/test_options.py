from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None,  required=True, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--Zpath', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='../Photo/0', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
		self.parser.add_argument('--num_style', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--interpolate', default=0, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--Gray', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Zlevel', default=9, type=int, help='Output size of generator')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--Notadd', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--remapping', action='store_true', help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--GfromZ', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--Metric', default=False, type=bool, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--CrossCat', default=0, type=float, help='Output size of generator')

		# arguments for style-mixing script
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='Downsampling factor for super-res (should be a single value for inference).')

	def parse(self):
		opts = self.parser.parse_args()
		return opts