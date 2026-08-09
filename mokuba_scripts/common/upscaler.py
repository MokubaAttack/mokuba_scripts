import os
import requests
import torch
import numpy
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

class mokuup:
	def __init__(self,path,dev):
		self.scale=2
		if path=="NEAREST":
			self.model=Image.NEAREST
			self.path="NEAREST"
			self.id=""
		elif path=="BOX":
			self.model=Image.BOX
			self.path="BOX"
			self.id=""
		elif path=="LANCZOS":
			self.model=Image.LANCZOS
			self.path="LANCZOS"
			self.id=""
		elif path=="HAMMING":
			self.model=Image.HAMMING
			self.path="HAMMING"
			self.id=""
		elif path=="BICUBIC":
			self.model=Image.BICUBIC
			self.path="BICUBIC"
			self.id=""
		elif path=="BILINEAR":
			self.model=Image.BILINEAR
			self.path="BILINEAR"
			self.id=""
		else:
			if not(os.path.exists(path)):
				path=os.getcwd()+'/upscaler/RealESRGAN_x4plus.pth'
				if not(os.path.exists(os.path.dirname(path))):
					os.mkdir(os.path.dirname(path))
				url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
				res = requests.get(url,stream=True)
				f=open(path, 'wb')
				for chunk in res.iter_content(chunk_size=1024*1024):
					f.write(chunk)
				f.close()
				del res

			sd=torch.load(path)
			if "id" in sd:
				self.id=str(sd["id"].item())
			else:
				self.id=str(164898)
			if "params" in sd:
				sd=sd["params"]
			elif "params_ema" in sd:
				sd=sd["params_ema"]
			else:
				self.model=Image.BILINEAR
				self.path="BILINEAR"
				self.id=""
				return
			if "conv_first.weight" in sd:
				nf=sd["conv_first.weight"].size()[0]
				for i in range(1000):
					if not("body."+str(i)+".rdb1.conv1.weight" in sd):
						break
				nb=i
				if 3*4==sd["conv_first.weight"].size()[1]:
					scale=2
				elif 3*16==sd["conv_first.weight"].size()[1]:
					scale=1
				else:
					scale=4
				net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=nf, num_block=nb, num_grow_ch=32, scale=scale)
			elif "body.0.weight" in sd:
				nf=sd["body.0.weight"].size()[0]
				for i in range(1000):
					if not("body."+str(i*2+4)+".weight" in sd):
						break
				nc=i
				for i in range(1,1000):
					if int(sd["body."+str(nc*2+2)+".weight"].size()[0])==int(3*(2**i)):
						break
				scale=i
				net=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=nf, num_conv=nc, upscale=scale, act_type='prelu')
			else:
				self.model=Image.BILINEAR
				self.path="BILINEAR"
				self.id=""
				return

			self.model = RealESRGANer(
				scale=scale,
				model_path=path,
				dni_weight=None,
				model=net,
				tile=256,
				tile_pad=10,
				pre_pad=0,
				half=True,
				device=dev
			)
			self.path=os.path.basename(path)
			self.scale=scale

	def get_method(self):
		return self.path,self.id

	def run(self,img,x,y):
		if not(self.path in ["NEAREST","BOX","LANCZOS","HAMMING","BICUBIC","BILINEAR"]):
			input_image = img.convert('RGB')
			input_image = numpy.array(input_image)
			while Image.fromarray(input_image).width<x or Image.fromarray(input_image).height<y:
				input_image, _ = self.model.enhance(input_image)
			input_image=Image.fromarray(input_image)
			if input_image.width==x and input_image.height==y:
				image0=input_image
			else:
				image0=input_image.resize((x,y))
		else:
			image0=img.resize((x,y), resample=self.model)
		return image0
		
	def get_scale(self):
		return self.scale