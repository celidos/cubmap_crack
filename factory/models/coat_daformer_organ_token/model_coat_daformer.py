
# +

from .daformer import *
from .coat import *
# -



# ################################################################

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
	IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
	
	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x - self.mean) / self.std
		return x


class MixUpSample(nn.Module):
	def __init__( self, scale_factor=2):
		super().__init__()
		self.mixing = nn.Parameter(torch.tensor(0.5))
		self.scale_factor = scale_factor
	
	def forward(self, x):
		x = self.mixing *F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
		    + (1-self.mixing )*F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
		return x


def AUXheadHR(in_channels, mid_channels, out_channels, upsample=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
        
        MixUpSample(scale_factor=upsample) if upsample > 1 else nn.Identity()
    )


def AUXheadLR(in_channels, mid_channels, out_channels, pix_shuffle=2, upsample=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels // 4, kernel_size=1, stride=1),
        nn.BatchNorm2d(mid_channels // 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels // 4, mid_channels, kernel_size=1, stride=1),
        nn.PixelShuffle(upscale_factor=pix_shuffle),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels // (pix_shuffle * pix_shuffle), out_channels, kernel_size=1, stride=1),
        MixUpSample(scale_factor=upsample)
    )


class Net(nn.Module):
	
	
	def __init__(self,
	             encoder=coat_lite_medium,
	             decoder=daformer_conv3x3,
	             encoder_cfg={},
	             decoder_cfg={},
                 n_classes=1,
	             ):
		super(Net, self).__init__()
		encoder_dim = encoder_cfg.get('encoder_dim', 320) 
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		self.n_classes = n_classes
		
		# ----
		self.rgb = RGB()
		
		self.encoder = encoder(
			#drop_path_rate=0.3,
		)
		encoder_dim = self.encoder.embed_dims
		# [64, 128, 320, 512]
		
		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, n_classes, kernel_size=1),
			MixUpSample(scale_factor=4)
		)
		self.aux_heads_serial = nn.ModuleList([
			AUXheadHR(in_channels=152, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
		])
		self.aux_heads_parallel = nn.ModuleList([
			AUXheadHR(in_channels=152, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
			AUXheadHR(in_channels=320, mid_channels=320, out_channels=n_classes, upsample=1),
		])
	 
	def forward(self, batch):
		organs = batch['organ']
        
        
		x = batch['image']

		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x, organs)
		output = dict()
		for i in range(len(encoder['parallel'])):
			output['aux_parallel_{}'.format(i)] = self.aux_heads_serial[i](encoder['parallel'][i])
		for i in range(len(encoder['serial'])):
			output['aux_serial_{}'.format(i)] = self.aux_heads_serial[i](encoder['serial'][i])
# 		print([f.shape for f in encoder])
		
		last, decoder = self.decoder(encoder['parallel'])
# 		print('before output shape', last.shape)
		logit = self.logit(last)
# 		print(logit.shape)
		
		output['logits'] = logit
# 		probability_from_logit = torch.sigmoid(logit)
# 		output['probability'] = probability_from_logit
		
		return output

 





def run_check_net():
	batch_size = 2
	image_size = 800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()
