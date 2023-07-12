from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torchvision


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		if(opts.sort):
			self.target_paths = sorted(data_utils.make_dataset(target_root))
		else:
			self.target_paths = data_utils.make_dataset(target_root)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')# if self.opts.label_nc == 0 else from_im.convert('L')
		
		if (self.opts.SS!=0):
			im90=torchvision.transforms.functional.rotate(from_im,90)
			im180=torchvision.transforms.functional.rotate(from_im,180)
			im270=torchvision.transforms.functional.rotate(from_im,270)
		# print(from_im.shape)
		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
			if(self.opts.SS!=0):
				im90=self.source_transform(im90)
				im180=self.source_transform(im180)
				im270=self.source_transform(im270)
				return from_im, to_im, im90, im180, im270
		else:
			from_im = to_im
		return from_im, to_im
