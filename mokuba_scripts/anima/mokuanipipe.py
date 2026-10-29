from diffusers import (
	AnimaModularPipeline,
	FlowMatchLCMScheduler,
	FlowMatchEulerDiscreteScheduler,
	CosmosTransformer3DModel
)
import torch
import os
from lycoris import create_lycoris_from_weights
from safetensors.torch import load_file

from .animackpt2 import (
	check_anima_base,
	safe2diff,
	folder2diff
)
from ..common.metadata import get_id
from ..common.flush import flush
from ..common.upscaler import mokuup
from ..common.lyco2 import lyco2anima

txt2img_params=["st","cf","pr","ne","se","ckpt","lora","w","sa"]
img2img_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","ds"]
hiresfix_params=["st","cf","pr","ne","se","ckpt","lora","w","sa","ds","up","hu","hum","hs"]

default_trans_config={
  "in_channels": 16,
  "out_channels": 16,
  "num_attention_heads": 16,
  "attention_head_dim": 128,
  "num_layers": 40,
  "mlp_ratio": 4.0,
  "text_embed_dim": 1024,
  "adaln_lora_dim": 256,
  "max_size": [
    128,
    240,
    240
  ],
  "patch_size": [
    1,
    2,
    2
  ],
  "rope_scale": [
    1.0,
    4.0,
    4.0
  ],
  "concat_padding_mask": True,
  "extra_pos_embed_type": None,
  "use_crossattn_projection": False,
  "crossattn_proj_in_channels": 1024,
  "encoder_hidden_states_channels": 1024,
  "controlnet_block_every_n": None,
  "img_context_dim_in": None,
  "img_context_num_tokens": 256,
  "img_context_dim_out": 2048,
  "_class_name": "CosmosTransformer3DModel",
  "_diffusers_version": "0.39.0.dev0",
  "_name_or_path": os.getcwd()+"/AnimaBaseV1"
}

class mokuanipipe:
	def __init__(self):
		self.moku_meta={
			"lora":[],
			"w":[],
			"embed":[],
			"sa":"FlowMatch_Euler normal"
		}

		check_anima_base()

		self.upscaler=None

	def from_safe(self,path,torch_dtype=torch.float,device="cpu"):
		self.moku_meta["ckpt"]=get_id(path)

		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,block_num=safe2diff(path=path)

		self.pipe=AnimaModularPipeline.from_pretrained(os.getcwd()+"/AnimaBaseV1")
		if transformer_sd!={}:
			transformer_config = default_trans_config
			transformer_config["num_layers"] = block_num
			self.pipe.transformer=CosmosTransformer3DModel.from_config(transformer_config)
			self.pipe.transformer.load_state_dict(transformer_sd)
			self.pipe.transformer.to(device)
		del transformer_sd
		self.pipe.load_components()
		self.pipe.to(device).to(torch_dtype)

		if text_encoder_sd!={}:
			for k,p in self.pipe.text_encoder.named_parameters():
				p.data=text_encoder_sd.pop(k).to(device).to(torch_dtype)
		del text_encoder_sd
		if text_conditioner_sd!={}:
			for k,p in self.pipe.text_conditioner.named_parameters():
				p.data=text_conditioner_sd.pop(k).to(device).to(torch_dtype)
		del text_conditioner_sd
		if vae_sd!={}:
			for k,p in self.pipe.vae.named_parameters():
				p.data=vae_sd.pop(k).to(device).to(torch_dtype)
		del vae_sd
		print(path+" is loaded.")
		flush()

	def from_folder(self,path,torch_dtype=torch.float,device="cpu"):
		self.moku_meta["ckpt"]=get_id(path)

		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,block_num=folder2diff(path=path)

		self.pipe=AnimaModularPipeline.from_pretrained(os.getcwd()+"/AnimaBaseV1")
		if transformer_sd!={}:
			transformer_config = default_trans_config
			transformer_config["num_layers"] = block_num
			self.pipe.transformer=CosmosTransformer3DModel.from_config(transformer_config)
			self.pipe.transformer.load_state_dict(transformer_sd)
			self.pipe.transformer.to(device)
		del transformer_sd
		self.pipe.load_components()
		self.pipe.to(device).to(torch_dtype)

		if text_encoder_sd!={}:
			for k,p in self.pipe.text_encoder.named_parameters():
				p.data=text_encoder_sd.pop(k).to(device).to(torch_dtype)
		del text_encoder_sd
		if text_conditioner_sd!={}:
			for k,p in self.pipe.text_conditioner.named_parameters():
				p.data=text_conditioner_sd.pop(k).to(device).to(torch_dtype)
		del text_conditioner_sd
		if vae_sd!={}:
			for k,p in self.pipe.vae.named_parameters():
				p.data=vae_sd.pop(k).to(device).to(torch_dtype)
		del vae_sd
		
		print(path+" is loaded.")
		flush()

	def set_scheduler(self,sample,sgm):
		if sgm.lower()=="karras":
			sgmuse=[True,False,False]
		elif sgm.lower()=="exponential":
			sgmuse=[False,True,False]
		elif sgm.lower()=="beta":
			sgmuse=[False,False,True]
		else:
			sgm="normal"
			sgmuse=[False,False,False]

		if sample.lower()=="FlowMatch_LCM".lower():
			self.pipe.scheduler=FlowMatchLCMScheduler.from_config(
				self.pipe.scheduler.config,
				use_karras_sigmas=sgmuse[0],
				use_exponential_sigmas=sgmuse[1],
				use_beta_sigmas=sgmuse[2],
			)
		else:
			sample="FlowMatch_Euler"
			self.pipe.scheduler=FlowMatchEulerDiscreteScheduler.from_config(
				self.pipe.scheduler.config,
				use_karras_sigmas=sgmuse[0],
				use_exponential_sigmas=sgmuse[1],
				use_beta_sigmas=sgmuse[2],
			)

		self.moku_meta["sa"]=sample+" "+sgm

	def load_lycoris(self,path,weight=1,trans=True,teco=True,teen=True):
		i,w=get_id(path,weight)
		self.moku_meta["lora"] += i
		self.moku_meta["w"] += w
		
		transformer_sd,text_conditioner_sd,text_encoder_sd=lyco2anima(path) 

		if transformer_sd=={} and text_encoder_sd=={} and text_conditioner_sd=={}:
			raise RuntimeWarning(path+" isn't supported.")
		if transformer_sd!={} and trans:
			wrapper, _ = create_lycoris_from_weights(multiplier=weight,file="dummy.safetensors",module=self.pipe.transformer, weights_sd=transformer_sd)
			wrapper.merge_to()
			del wrapper
		del transformer_sd
		flush()
		if text_encoder_sd!={} and teen:
			wrapper, _ = create_lycoris_from_weights(multiplier=weight,file="dummy.safetensors",module=self.pipe.text_encoder, weights_sd=text_encoder_sd)
			wrapper.merge_to()
			del wrapper
		del text_encoder_sd
		flush()
		if text_conditioner_sd!={} and teco:
			wrapper, _ = create_lycoris_from_weights(multiplier=weight,file="dummy.safetensors",module=self.pipe.text_conditioner, weights_sd=text_conditioner_sd)
			wrapper.merge_to()
			del wrapper
		del text_conditioner_sd
		flush()
		print(path+" is loaded.")

	def make_upscaler(self,path="BILINEAR",dev="cpu"):
		self.upscaler=mokuup(path,dev)
		self.moku_meta["hum"],self.moku_meta["up"]=self.upscaler.get_method()

	def upscale(self,img,x,y):
		if self.upscaler==None:
			raise RuntimeWarning("You must make a upscaler.")
		self.moku_meta["hu"]=str(x/img.width)
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
		seed
		):
		self.pipe.guider.register_to_config(guidance_scale=gs)
		self.moku_meta["st"]=str(step)
		self.moku_meta["cf"]=str(gs)
		self.moku_meta["pr"]=prompt
		self.moku_meta["ne"]=n_prompt
		self.moku_meta["se"]=str(seed)
		self.pipe.vae.enable_tiling()

		out_img = self.pipe(
			prompt=prompt,
			negative_prompt=n_prompt,
			num_inference_steps=step,
			generator=torch.manual_seed(seed),
			width=x,
			height=y
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
		hiresfix=False
		):
		self.pipe.guider.register_to_config(guidance_scale=gs)
		if hiresfix and "st" in self.moku_meta:
			self.moku_meta["hs"]=str(step)
		else:
			self.moku_meta["st"]=str(step)
		self.moku_meta["ds"]=str(ss)
		self.moku_meta["cf"]=str(gs)
		self.moku_meta["pr"]=prompt
		self.moku_meta["ne"]=n_prompt
		self.moku_meta["se"]=str(seed)
		self.pipe.vae.enable_tiling()

		out_img = self.pipe(
			prompt=prompt,
			negative_prompt=n_prompt,
			num_inference_steps=int(step/ss)+1,
			generator=torch.manual_seed(seed),
			width=img.width,
			height=img.height,
			strength=ss,
			image=img
		).images[0]

		out_meta={}
		if hiresfix:
			for k in hiresfix_params:
				out_meta[k]=self.moku_meta.get(k,"")
		else:
			for k in img2img_params:
				out_meta[k]=self.moku_meta.get(k,"")

		return out_img,out_meta
