
from daformer import *
from coat import *



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


def AUXhead(in_channels, mid_channels, out_channels, upsample=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=upsample, mode='bilinear')
    )


def AUXheadHR(in_channels, mid_channels, out_channels, upsample=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        MixUpSample(scale_factor=upsample)
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


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


# +
class UnetDecoder(nn.Module):
    def __init__(self):
        # params are ignored
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 72, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.n_channels = 320
        self.size = [320, 320, 320, 152, 64, 32]
        self.out_channels = 6

        self.up_conv1 = up_conv(320, 320)
        self.double_conv1 = double_conv(320 + 320, 320)
        self.up_conv2 = up_conv(320, 320)
        self.double_conv2 = double_conv(320 + 320, 320)
        self.up_conv3 = up_conv(320, 320)
        self.double_conv3 = double_conv(320 + 152, 320)
        self.up_conv4 = up_conv(320, 320)
        self.double_conv4 = double_conv(320 + 72, 256)

#         if self.concat_input:
        self.up_conv_input = up_conv(256, 256)
        self.double_conv_input = double_conv(256 + 3, 256)

        self.final_conv = nn.Conv2d(256, 6, kernel_size=1)

    def forward(self, batch, features):
        features = features[::-1]
        img = batch['image']
        
        conv_features = self.init_conv(img)

#         print(conv_features.shape)
#         for el in features:
#             print(el.shape)
#         print('x', features[0].shape)
        x = self.up_conv1(features[0])
#         print('xx', x.shape)
        x = torch.cat([x, features[1]], dim=1)
        x = self.double_conv1(x)
#         print('x dc1', x.shape)

        x = self.up_conv2(x)
        x = torch.cat([x, features[2]], dim=1)
        aux2 = self.double_conv2(x)
        x = aux2
#         print('x dc2', x.shape)

        x = self.up_conv3(x)
        x = torch.cat([x, features[3]], dim=1)
        aux3 = self.double_conv3(x)
        x = aux3
#         print('x dc3', x.shape)

        x = self.up_conv4(x)
        x = torch.cat([x, conv_features], dim=1)
        aux4 = self.double_conv4(x)
        x = aux4
#         print('x dc4', x.shape)
        
        x = self.up_conv_input(x)
        x = torch.cat([x, img], dim=1)
        x = self.double_conv_input(x)

        x = self.final_conv(x)

        return {'x': x, 'aux4': aux4, 'aux3': aux3, 'aux2': aux2}


# +
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
# 			encoder_dim=encoder_dim,
# 			decoder_dim=decoder_dim,
		)
# 		self.logit = nn.Sequential(
# 			nn.Conv2d(decoder_dim, 1024, kernel_size=1),
# 			nn.GELU(),
# 			nn.Dropout(p=0.05),
# 			nn.PixelShuffle(upscale_factor=2),
# 			MixUpSample(scale_factor=2),
# 			nn.Conv2d(256, n_classes, kernel_size=1),
# 		)
		self.aux_heads_serial = nn.ModuleList([
			AUXheadHR(in_channels=152, mid_channels=256, out_channels=n_classes, upsample=4),
			AUXheadHR(in_channels=320, mid_channels=256, out_channels=n_classes, upsample=8),
			AUXheadLR(in_channels=320, mid_channels=512, out_channels=n_classes, pix_shuffle=2, upsample=8),
			AUXheadLR(in_channels=320, mid_channels=1024, out_channels=n_classes, pix_shuffle=4, upsample=8),
		])
		self.aux_heads_parallel = nn.ModuleList([
			AUXheadHR(in_channels=152, mid_channels=256, out_channels=n_classes, upsample=4),
			AUXheadHR(in_channels=320, mid_channels=256, out_channels=n_classes, upsample=8),
			AUXheadLR(in_channels=320, mid_channels=512, out_channels=n_classes, pix_shuffle=2, upsample=8),
			AUXheadLR(in_channels=320, mid_channels=1024, out_channels=n_classes, pix_shuffle=4, upsample=8),
		])
        
		self.aux3head = AUXhead(320, 256, n_classes, 4)
		self.aux2head = AUXhead(320, 320, n_classes, 8)
	 
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x)
		output = dict()
		for i in range(len(encoder['parallel'])):
			output['aux_parallel_{}'.format(i)] = self.aux_heads_serial[i](encoder['parallel'][i])
		for i in range(len(encoder['serial'])):
			output['aux_serial_{}'.format(i)] = self.aux_heads_serial[i](encoder['serial'][i])
# 		print([f.shape for f in encoder])
		
		decode_result = self.decoder(batch, encoder['parallel'])
		logit = decode_result['x']
# 		print('before output shape', last.shape)
# 		logit = self.logit(last)
# 		print(logit.shape)
		
		output['logits'] = logit
		output['aux2_unet'] = self.aux2head(decode_result['aux2'])
		output['aux3_unet'] = self.aux3head(decode_result['aux3'])
        
# 		probability_from_logit = torch.sigmoid(logit)
# 		output['probability'] = probability_from_logit
		
		return output
# -

 





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
