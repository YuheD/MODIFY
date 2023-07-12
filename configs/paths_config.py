dataset_paths = {
	'celeba_train': '/data/yuhe.ding/code/DATABASE/Met-clean',
	'celeba_test': '/data/yuhe.ding/code/DATABASE/Met-clean',
	'celeba_train_sketch': '/data1/yuhe.ding/code/DATABASE/Met-clean',
	'celeba_test_sketch': '/data1/yuhe.ding/code/DATABASE/Met-clean',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
}
DA_dataset_paths = {
	'celeba_train': '/data/yuhe.ding/DATA/Photo',
	'celeba_test': '/data/yuhe.ding/DATA/Photo',
	'celeba_train_sketch': '/data/yuhe.ding/DATA/Photo',
	'celeba_test_sketch': '/data/yuhe.ding/DATA/Photo',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
