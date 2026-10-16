from diffusers import (
	AutoencoderKL,
	ControlNetModel,
	StableDiffusionPipeline,
	StableDiffusionImg2ImgPipeline,
	StableDiffusionControlNetImg2ImgPipeline,
	EulerDiscreteScheduler,
	EulerAncestralDiscreteScheduler,
	LMSDiscreteScheduler,
	HeunDiscreteScheduler,
	KDPM2DiscreteScheduler,
	KDPM2AncestralDiscreteScheduler,
	DPMSolverMultistepScheduler,
	DPMSolverSinglestepScheduler,
	PNDMScheduler,
	UniPCMultistepScheduler,
	LCMScheduler,
	DDIMScheduler,
	DPMSolverSDEScheduler
)
import torch
import os
from lycoris import create_lycoris_from_weights
from safetensors.torch import load_file
from compel import CompelForSD
import numpy
from optimum.quanto import (
	freeze,
	qfloat8,
	quantize
)
import cv2
from PIL import Image

from ..common.metadata import get_id
from ..common.flush import flush
from ..common.upscaler import mokuup
from ..common.lyco2 import lyco2sd

sgm_use=[
	"Euler","Euler a","DPM++ 2M","DPM++ 2M SDE","DPM++ SDE","DPM++","DPM2","DPM2 a","Heun","LMS","UniPC","DPM++ 3M SDE"
]

txt2img_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","cl","embed","vae"]
img2img_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","ds","cl","embed","vae"]
hiresfix_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","ds","up","hu","hum","hs","cl","embed","vae"]
tileup_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","ds","up","tu","tum","cl","embed","vae","ccs","cont"]

def create_gaussian_weight(w,h, sigma=0.3):
	x = numpy.linspace(-1, 1, w)
	y = numpy.linspace(-1, 1, h)
	xx, yy = numpy.meshgrid(x, y)
	gaussian_weight = numpy.exp(-(xx**2 + yy**2) / (2 * sigma**2))
	return gaussian_weight

class mokusdpipe:
	def __init__(self):
		self.moku_meta={
			"lora":[],
			"w":[],
			"embed":[],
			"sa":"DDIM"
		}

		self.prompt_plus=""
		self.n_prompt_plus=""

		self.upscaler=None
		self.pipe=None

	def from_safe(self,path,torch_dtype=torch.float,device="cpu",vae_path="vae.safetensors"):
		self.moku_meta["ckpt"]=get_id(path)

		self.pipe=StableDiffusionPipeline.from_single_file(path,torch_dtype=torch_dtype)
		self.pipe.to(device)
		print(path+" is loaded.")
		
		self.dtype=torch_dtype
		self.dev=device

		if os.path.exists(vae_path):
			self.moku_meta["vae"]=get_id(vae_path)
			self.pipe.vae=AutoencoderKL.from_single_file(vae_path,torch_dtype=torch_dtype)
			print(vae_path+" is loaded.")

	def from_folder(self,path,torch_dtype=torch.float,device="cpu",vae_path="vae.safetensors"):
		self.moku_meta["ckpt"]=get_id(path)

		self.pipe=StableDiffusionPipeline.from_pretrained(path,torch_dtype=torch_dtype)
		self.pipe.to(device)
		print(path+" is loaded.")
		
		self.dtype=torch_dtype
		self.dev=device

		if os.path.exists(vae_path):
			self.moku_meta["vae"]=get_id(vae_path)
			self.pipe.vae=AutoencoderKL.from_single_file(vae_path,torch_dtype=torch_dtype)
			print(vae_path+" is loaded.")

	def set_scheduler(self,sample,sgm):
		self.moku_meta["sa"]=""
		sgm_dict={}
		sgm_dict["use_karras_sigmas"]=False
		sgm_dict["use_exponential_sigmas"]=False
		sgm_dict["use_beta_sigmas"]=False
		if sample in sgm_use:
			if sgm=="Karras":
				sgm_dict["timestep_spacing"]="linspace"
				sgm_dict["use_karras_sigmas"]=True
				self.moku_meta["sa"]=" "+sgm
			elif sgm=="exponential":
				sgm_dict["timestep_spacing"]="linspace"
				sgm_dict["use_exponential_sigmas"]=True
				self.moku_meta["sa"]=" "+sgm
			elif sgm=="beta":
				sgm_dict["timestep_spacing"]="linspace"
				sgm_dict["use_beta_sigmas"]=True
				self.moku_meta["sa"]=" "+sgm
			elif sgm=="sgm_uniform" or sgm=="simple":
				sgm_dict["timestep_spacing"]="trailing"
				self.moku_meta["sa"]=" "+sgm
			else:
				sgm_dict["timestep_spacing"]="linspace"
				self.moku_meta["sa"]=" "+sgm
		else:
			if sgm=="sgm_uniform" or sgm=="simple":
				sgm_dict["timestep_spacing"]="trailing"
				self.moku_meta["sa"]=" "+sgm
			else:
				sgm_dict["timestep_spacing"]="leading"
				self.moku_meta["sa"]=" "+sgm

		if sample=="Euler":
			self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="Euler a":
			self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="LMS":
			self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="Heun":
			self.pipe.scheduler = HeunDiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM2":
			self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM2 a":
			self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM++ 2M":
			self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM++ SDE":
			self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
			algorithm_type="sde-dpmsolver++",
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM++":
			self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM++ 2M SDE":
			self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config,
			algorithm_type="sde-dpmsolver++",
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="PLMS":
			self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="UniPC":
			self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="LCM":
			self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		elif sample=="DPM++ 3M SDE":
			self.pipe.scheduler = DPMSolverSDEScheduler.from_config(self.pipe.scheduler.config,
			timestep_spacing=sgm_dict["timestep_spacing"],
			use_karras_sigmas=sgm_dict["use_karras_sigmas"],
			use_exponential_sigmas=sgm_dict["use_exponential_sigmas"],
			use_beta_sigmas=sgm_dict["use_beta_sigmas"]
			)
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]
		else:
			self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config,timestep_spacing=sgm_dict["timestep_spacing"])
			self.moku_meta["sa"]=sample+self.moku_meta["sa"]

	def load_lycoris(self,path,weight=1,unet=True,teen1=True):
		i,w=get_id(path,weight)
		self.moku_meta["lora"] += i
		self.moku_meta["w"] += w
		
		unet_sd,text_encoder_sd=lyco2sd(path)

		if unet_sd=={} and text_encoder_sd=={}:
			raise RuntimeWarning(path+" isn't supported.")
		if unet_sd!={} and unet:
			wrapper, _ = create_lycoris_from_weights(multiplier=weight,file="dummy.safetensors",module=self.pipe.unet, weights_sd=unet_sd)
			wrapper.merge_to()
			del wrapper
		del unet_sd
		flush()
		if text_encoder_sd!={} and teen1:
			wrapper, _ = create_lycoris_from_weights(multiplier=weight,file="dummy.safetensors",module=self.pipe.text_encoder, weights_sd=text_encoder_sd)
			wrapper.merge_to()
			del wrapper
		del text_encoder_sd
		flush()
		print(path+" is loaded.")

	def load_pos_embed(self,path):
		i,_=get_id(path,1)
		self.moku_meta["embed"] += i

		key=os.path.basename(path).removesuffix(".safetensors")
		self.pipe.load_textual_inversion(".", weight_name=path, token=key)

		self.prompt_plus+=", "+key
		print(path+" is loaded.")

	def load_neg_embed(self,path):
		i,_=get_id(path,1)
		self.moku_meta["embed"] += i

		key=os.path.basename(path).removesuffix(".safetensors")
		self.pipe.load_textual_inversion(".", weight_name=path, token=key)

		self.n_prompt_plus+=", "+key
		print(path+" is loaded.")

	def set_freeze_unet(self):
		quantize(self.pipe.unet, weights=qfloat8)
		freeze(self.pipe.unet)

	def set_gpu_lowmem(self):
		self.pipe.vae.tile_sample_min_size=256
		sample_size=256
		self.pipe.vae.tile_latent_min_size = int(sample_size / (2 ** (len(self.pipe.vae.config.block_out_channels) - 1)))
		self.pipe.enable_model_cpu_offload()

	def make_upscaler(self,path="BILINEAR",dev="cpu"):
		self.upscaler=mokuup(path,dev)
		self.moku_meta["hum"],self.moku_meta["up"]=self.upscaler.get_method()
		self.moku_meta["tum"],self.moku_meta["up"]=self.upscaler.get_method()

	def upscale(self,img,x,y):
		if self.upscaler==None:
			raise RuntimeWarning("You must make a upscaler.")
		self.moku_meta["hu"]=str(x/img.width)
		self.moku_meta["tu"]=str(x/img.width)
		out_img=self.upscaler.run(img,x,y)
		return out_img

	def delete_upscaler(self):
		if self.upscaler!=None:
			del self.upscaler
			self.upscaler=None
		flush()

	def txt2img(
		self,
		prompt,
		n_prompt,
		gs,
		step,
		x,
		y,
		seed,
		cs
		):
		self.pipe=StableDiffusionPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		self.pipe.to(self.dev)

		self.moku_meta["st"]=str(step)
		self.moku_meta["cf"]=str(gs)
		self.moku_meta["pr"]=prompt+self.prompt_plus
		self.moku_meta["ne"]=n_prompt+self.n_prompt_plus
		self.moku_meta["se"]=str(seed)
		self.moku_meta["cl"]=str(cs)
		self.pipe.vae.enable_tiling()

		compel = CompelForSD(self.pipe)
		conditioning = compel(prompt+self.prompt_plus, negative_prompt=n_prompt+self.n_prompt_plus)
		prompts=[conditioning.embeds,conditioning.negative_embeds]

		out_img = self.pipe(
			eta=1.0,
			prompt_embeds=prompts[0],
			negative_prompt_embeds=prompts[1],
			guidance_scale=gs,
			num_inference_steps=step,
			generator=torch.manual_seed(seed),
			width=x,
			height=y,
			clip_skip=cs
		).images[0]

		out_meta={}
		for k in txt2img_params:
			out_meta[k]=self.moku_meta.get(k,"")

		return out_img,out_meta

	def img2img(
		self,
		prompt,
		n_prompt,
		gs,
		step,
		ss,
		img,
		seed,
		cs,
		hiresfix=False
		):
		self.pipe=StableDiffusionImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
		self.pipe.to(self.dev)

		if hiresfix and "st" in self.moku_meta:
			self.moku_meta["hs"]=str(step)
		else:
			self.moku_meta["st"]=str(step)
		self.moku_meta["ds"]=str(ss)
		self.moku_meta["cf"]=str(gs)
		self.moku_meta["pr"]=prompt+self.prompt_plus
		self.moku_meta["ne"]=n_prompt+self.n_prompt_plus
		self.moku_meta["se"]=str(seed)
		self.moku_meta["cl"]=str(cs)
		self.pipe.vae.enable_tiling()

		compel = CompelForSD(self.pipe)
		conditioning = compel(prompt+self.prompt_plus, negative_prompt=n_prompt+self.n_prompt_plus)
		prompts=[conditioning.embeds,conditioning.negative_embeds]

		out_img = self.pipe(
			eta=1.0,
			prompt_embeds=prompts[0],
			negative_prompt_embeds=prompts[1],
			num_inference_steps=int(step/ss)+1,
			generator=torch.manual_seed(seed),
			width=img.width,
			height=img.height,
			strength=ss,
			image=img,
			guidance_scale=gs,
			clip_skip=cs
		).images[0]

		out_meta={}
		if hiresfix:
			for k in hiresfix_params:
				out_meta[k]=self.moku_meta.get(k,"")
		else:
			for k in img2img_params:
				out_meta[k]=self.moku_meta.get(k,"")

		return out_img,out_meta

	def tileup(
		self,
		prompt,
		n_prompt,
		gs,
		step,
		cs,
		seed,
		ss,
		img,
		ccs=0
		):
		if ccs==0:
			self.pipe=StableDiffusionImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype)
			self.moku_meta["ccs"]=""
			self.moku_meta["cont"]=""
		else:
			controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',torch_dtype=self.dtype)
			self.pipe=StableDiffusionControlNetImg2ImgPipeline.from_pipe(self.pipe,torch_dtype=self.dtype,controlnet=controlnet)
			self.moku_meta["ccs"]=str(ccs)
			self.moku_meta["cont"]=str(67566)
		self.pipe.to(self.dev)

		self.moku_meta["st"]=str(step)
		self.moku_meta["ds"]=str(ss)
		self.moku_meta["cf"]=str(gs)
		self.moku_meta["pr"]=prompt+self.prompt_plus
		self.moku_meta["ne"]=n_prompt+self.n_prompt_plus
		self.moku_meta["se"]=str(seed)
		self.moku_meta["cl"]=str(cs)
		self.pipe.vae.enable_tiling()

		compel = CompelForSD(self.pipe)
		conditioning = compel(prompt+self.prompt_plus, negative_prompt=n_prompt+self.n_prompt_plus)
		prompts=[conditioning.embeds,conditioning.negative_embeds]
		
		x=img.width
		y=img.height
		aspect_ratio = x/y
		if aspect_ratio>1:
			tile_w = min(x, 2*512)
			tile_h = min(round(tile_w /aspect_ratio/8)*8, 2*512)
		else:
			tile_h = min(y, 2*512)
			tile_w = min(round(tile_h*aspect_ratio/8)*8, 2*512)
		tile_w = max(512,tile_w)
		tile_h = max(512,tile_h)
		overlap = min(64, tile_w // 8, tile_h // 8)

		result = numpy.zeros((y, x, 3), dtype=numpy.float32)
		weight_sum = numpy.zeros((y, x, 1), dtype=numpy.float32)
		gaussian_weight = create_gaussian_weight(tile_w,tile_h,0.3)

		bottom=overlap
		while bottom<y:
			right=overlap
			top=bottom-overlap
			bottom=min(top+tile_h,y)
			while right<x:
				left=right-overlap
				right=min(left+tile_w,x)
				current_tile_size = (right - left,bottom - top)

				tile = img.crop((left, top, right, bottom))
				if ccs==0:
					result_tile = self.pipe(
						eta=1.0,
						prompt_embeds=prompts[0],
						negative_prompt_embeds=prompts[1],
						image=tile,
						guidance_scale=gs,
						generator=torch.manual_seed(seed),
						num_inference_steps=int(step/ss)+1,
						clip_skip=cs,
						strength=ss,
					).images[0]
				else:
					result_tile = self.pipe(
						eta=1.0,
						prompt_embeds=prompts[0],
						negative_prompt_embeds=prompts[1],
						image=tile,
						control_image=tile,
						guidance_scale=gs,
						generator=torch.manual_seed(seed),
						num_inference_steps=int(step/ss)+1,
						clip_skip=cs,
						strength=ss,
						controlnet_conditioning_scale=ccs,
					).images[0]

				if current_tile_size!=(result_tile.width,result_tile.height):
					result_tile = result_tile.resize( current_tile_size)

				if current_tile_size != (tile_w, tile_h):
					tile_weight = cv2.resize(gaussian_weight,current_tile_size)
				else:
					tile_weight = gaussian_weight[:current_tile_size[1], :current_tile_size[0]]

				numpy_result_tile = numpy.array(result_tile)
				result[top:bottom,left:right]=result[top:bottom,left:right]+numpy_result_tile*tile_weight[:,:,numpy.newaxis]
				weight_sum[top:bottom,left:right]=weight_sum[top:bottom,left:right]+tile_weight[:,:,numpy.newaxis]
				del tile_weight,result_tile,tile,numpy_result_tile
				flush()
		final_result = (result / weight_sum).astype(numpy.uint8)
		out_img = Image.fromarray(final_result)
		del final_result,result,weight_sum

		out_meta={}
		for k in tileup_params:
			out_meta[k]=self.moku_meta.get(k,"")

		return out_img,out_meta
