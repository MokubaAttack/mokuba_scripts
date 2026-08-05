import re
import torch
import os
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline

from .keys import (
	anima,
	sdxl
)

class diff2anima:
	def trans(self,key):
		block_maps_swap = {v: k for k, v in anima.block_maps.items()}
		root_map_swap = {v: k for k, v in anima.root_map.items()}

		block_re = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
		m = block_re.match(key)
		if m==None:
			for k2 in root_map_swap:
				key=key.replace(k2,root_map_swap[k2])
			key="model.diffusion_model."+key
		else:
			block_index = m.group(1)
			tail = m.group(2)
			mapped_tail = block_maps_swap.get(tail)
			key="model.diffusion_model.blocks."+block_index+"."+mapped_tail
		return key

	def teco(self,key):
		return "model.diffusion_model.llm_adapter."+key

	def teen(self,key):
		return "cond_stage_model.qwen3_06b.transformer.model."+key

	def vae(self,key):
		vae_keys1_swap={v: k for k, v in reversed(anima.vae_keys1.items())}
		vae_keys2_swap={v: k for k, v in reversed(anima.vae_keys2.items())}
		vae_keys3_swap={v: k for k, v in reversed(anima.vae_keys3.items())}

		if key.startswith("encoder."):
			for k2 in vae_keys2_swap:
				key=key.replace(k2,vae_keys2_swap[k2])
		elif key.startswith("decoder."):
			for k2 in vae_keys3_swap:
				key=key.replace(k2,vae_keys3_swap[k2])
		else:
			for k2 in vae_keys1_swap:
				key=key.replace(k2,vae_keys1_swap[k2])
		return "first_stage_model."+key

	def lora(self,path):
		sd=load_file(path)
		mappings={}
		for k in sd:
			if not(k.endswith((".lora_A.weight",".lora_down.weight"))):
				continue
			if ".lora_A.weight" in k:
				k=k.removesuffix(".lora_A.weight")
			else:
				k=k.removesuffix(".lora_down.weight")
			key=k.replace(".","_")
			m=re.search(r"layers_([0-9]+)_(.+)$",key)
			if m!=None:
				key="lora_te_"+m.group()
				mappings[k]=key
				continue

			m=re.search(r"llm_adapter_blocks(.+)$",key)
			if m!=None:
				key="lora_unet_"+m.group()
				mappings[k]=key
				continue

			m=re.search(r"blocks_[0-9]+_(.+)$",key)
			if m!=None:
				key=m.group()
				for k2 in anima.block_maps:
					mk2=k2.removesuffix(".weight").replace(".","_")
					mk2_value=anima.block_maps[k2].removesuffix(".weight").replace(".","_")
					key=key.replace(mk2_value,mk2)
				mappings[k]="lora_unet_"+key
				continue

			for k2 in anima.root_map:
				mk2=k2.removesuffix(".weight").replace(".","_")
				mk2_value=anima.root_map[k2].removesuffix(".weight").replace(".","_")
				m=re.search(mk2,key)
				if m!=None:
					key=mk2
					mappings[k]="lora_unet_"+key
					continue
				m=re.search(mk2_value,key)
				if m!=None:
					key=mk2
					mappings[k]="lora_unet_"+key
					continue

		sd_out={}
		endkeys=(".lora_A.weight",".lora_B.weight",".lora_down.weight",".lora_up.weight",".alpha")
		for key in mappings:
			for k in endkeys:
				if key+k in sd:
					w=sd.pop(key+k)
					if k==".lora_down.weight":
						k=".lora_A.weight"
					elif k==".lora_up.weight":
						k=".lora_B.weight"
					sd_out[mappings[key]+k]=w

		f=open(path+".txt","w")
		for k in sd:
			f.write(k+"\n")
		f.close()
		return sd_out

class diff2sdxl:
	def pipe(self,pipe):
		unet_conversion_map_swap={b:a for a,b in sdxl.unet_conversion_map.items()}
		unet_sd={}
		for k,p in pipe.unet.named_parameters():
			if k.removesuffix("weight") in unet_conversion_map_swap:
				unet_sd["model.diffusion_model."+unet_conversion_map_swap[k.removesuffix("weight")]+"weight"]=p.data
				continue
			elif k.removesuffix("bias") in unet_conversion_map_swap:
				unet_sd["model.diffusion_model."+unet_conversion_map_swap[k.removesuffix("bias")]+"bias"]=p.data
				continue
			mkey=k
			if "resnets" in k:
				for k2 in sdxl.unet_conversion_map_resnet:
					mkey = mkey.replace(sdxl.unet_conversion_map_resnet[k2], k2)
			for k2 in sdxl.unet_conversion_map_layer:
				mkey = mkey.replace(k2[1], k2[0])
			unet_sd["model.diffusion_model."+mkey]=p.data
			
		teen1_sd={}
		for k,p in pipe.text_encoder.named_parameters():
			teen1_sd["conditioner.embedders.0.transformer."+k]=p.data
			
		textenc_root_map_swap={b:a for a,b in sdxl.textenc_root_map.items()}
		teen2_sd={}
		weight_dict={}
		bias_dict={}
		for k,p in pipe.text_encoder_2.named_parameters():
			if k in textenc_root_map_swap:
				for k2 in textenc_root_map_swap:
					k=k.replace(k2,textenc_root_map_swap[k2])
				if k=="text_projection":
					teen2_sd[k]=p.data.T.contiguous()
				else:
					teen2_sd[k]=p.data
				continue
			
			mkey=k
			for k2 in sdxl.textenc_conversion_lst:
				mkey=mkey.replace(sdxl.textenc_conversion_lst[k2],k2)

			if mkey.endswith(".attn.q_proj.weight") or mkey.endswith(".attn.k_proj.weight") or mkey.endswith(".attn.v_proj.weight"):
				if mkey.endswith(".attn.q_proj.weight"):
					wkey=mkey.removesuffix(".q_proj.weight")
					wind=0
				elif mkey.endswith(".attn.k_proj.weight"):
					wkey=mkey.removesuffix(".k_proj.weight")
					wind=1
				else:
					wkey=mkey.removesuffix(".v_proj.weight")
					wind=2
				if not(wkey in weight_dict):
					weight_dict[wkey]=[None,None,None]
				weight_dict[wkey][wind]=p.data
				if weight_dict[wkey][0]!=None and weight_dict[wkey][1]!=None and weight_dict[wkey][2]!=None:
					teen2_sd["conditioner.embedders.1.model." +wkey+".in_proj_weight"]=torch.cat(weight_dict[wkey])
				continue
			if bkey.endswith(".attn.q_proj.bias") or mkey.endswith(".attn.k_proj.bias") or mkey.endswith(".attn.v_proj.bias"):
				if mkey.endswith(".attn.q_proj.bias"):
					bkey=mkey.removesuffix(".q_proj.bias")
					bind=0
				elif mkey.endswith(".attn.k_proj.bias"):
					bkey=mkey.removesuffix(".k_proj.bias")
					bind=1
				else:
					bkey=mkey.removesuffix(".v_proj.bias")
					bind=2
				if not(bkey in bias_dict):
					bias_dict[bkey]=[None,None,None]
				bias_dict[bkey][bind]=p.data
				if bias_dict[bkey][0]!=None and bias_dict[bkey][1]!=None and bias_dict[bkey][2]!=None:
					teen2_sd["conditioner.embedders.1.model." +wkey+".in_proj_bias"]=torch.cat(bias_dict[bkey])
				continue
			teen2_sd["conditioner.embedders.1.model." +mkey]=p.data

		vae_sd={}
		for k,p in pipe.vae.named_parameters():
			mkey=k
			for k2 in sdxl.vae_conversion_map:
				mkey=mkey.replace(sdxl.vae_conversion_map[k2],k2)
			if "attentions" in k:
				for k2 in sdxl.vae_conversion_map_attn:
					mkey = mkey.replace(sdxl.vae_conversion_map_attn[k2], k2)
			for k2 in ["q", "k", "v", "proj_out"]:
				if "mid.attn_1."+k2+".weight" in mkey:
					if p.data.ndim != 1:
						p.data=p.data.reshape(*p.data.shape,1,1)
			vae_sd["first_stage_model."+mkey]=p.data
		return {**unet_sd,**teen1_sd,**teen2_sd,**vae_sd}

	def folder(self,path):
		unet_path = path+"/unet/diffusion_pytorch_model.safetensors"
		vae_path = path+"/vae/diffusion_pytorch_model.safetensors"
		teen_path = path+"/text_encoder/model.safetensors"
		teen2_path = path+"/text_encoder_2/model.safetensors"

		if os.path.exists(unet_path):
			unet_sd_org=load_file(unet_path)
		else:
			unet_path = path+"/unet/diffusion_pytorch_model.bin"
			unet_sd_org=torch.load(unet_path)
		unet_conversion_map_swap={b:a for a,b in sdxl.unet_conversion_map.items()}
		unet_sd={}
		for k,p in unet_sd_org.items():
			if k.removesuffix("weight") in unet_conversion_map_swap:
				unet_sd["model.diffusion_model."+unet_conversion_map_swap[k.removesuffix("weight")]+"weight"]=p
				continue
			elif k.removesuffix("bias") in unet_conversion_map_swap:
				unet_sd["model.diffusion_model."+unet_conversion_map_swap[k.removesuffix("bias")]+"bias"]=p
				continue
			mkey=k
			if "resnets" in k:
				for k2 in sdxl.unet_conversion_map_resnet:
					mkey = mkey.replace(sdxl.unet_conversion_map_resnet[k2], k2)
			for k2 in sdxl.unet_conversion_map_layer:
				mkey = mkey.replace(k2[1], k2[0])
			unet_sd["model.diffusion_model."+mkey]=p
		del unet_sd_org
			
		if os.path.exists(teen_path):
			teen_sd_org=load_file(teen_path)
		else:
			teen_path = path+"/text_encoder/pytorch_model.bin"
			teen_sd_org=torch.load(teen_path)
		teen1_sd={}
		for k,p in teen_sd_org.items():
			teen1_sd["conditioner.embedders.0.transformer."+k]=p
		del teen_sd_org
			
		if os.path.exists(teen2_path):
			teen2_sd_org=load_file(teen_path)
		else:
			teen2_path = path+"/text_encoder_2/pytorch_model.bin"
			teen2_sd_org=torch.load(teen2_path)
		textenc_root_map_swap={b:a for a,b in sdxl.textenc_root_map.items()}
		teen2_sd={}
		weight_dict={}
		bias_dict={}
		for k,p in teen2_sd_org.items():
			if k in textenc_root_map_swap:
				for k2 in textenc_root_map_swap:
					k=k.replace(k2,textenc_root_map_swap[k2])
				if k=="text_projection":
					teen2_sd[k]=p.T.contiguous()
				else:
					teen2_sd[k]=p
				continue
			
			mkey=k
			for k2 in sdxl.textenc_conversion_lst:
				mkey=mkey.replace(sdxl.textenc_conversion_lst[k2],k2)

			if mkey.endswith(".attn.q_proj.weight") or mkey.endswith(".attn.k_proj.weight") or mkey.endswith(".attn.v_proj.weight"):
				if mkey.endswith(".attn.q_proj.weight"):
					wkey=mkey.removesuffix(".q_proj.weight")
					wind=0
				elif mkey.endswith(".attn.k_proj.weight"):
					wkey=mkey.removesuffix(".k_proj.weight")
					wind=1
				else:
					wkey=mkey.removesuffix(".v_proj.weight")
					wind=2
				if not(wkey in weight_dict):
					weight_dict[wkey]=[None,None,None]
				weight_dict[wkey][wind]=p
				if weight_dict[wkey][0]!=None and weight_dict[wkey][1]!=None and weight_dict[wkey][2]!=None:
					teen2_sd["conditioner.embedders.1.model." +wkey+".in_proj_weight"]=torch.cat(weight_dict[wkey])
				continue

			if bkey.endswith(".attn.q_proj.bias") or mkey.endswith(".attn.k_proj.bias") or mkey.endswith(".attn.v_proj.bias"):
				if mkey.endswith(".attn.q_proj.bias"):
					bkey=mkey.removesuffix(".q_proj.bias")
					bind=0
				elif mkey.endswith(".attn.k_proj.bias"):
					bkey=mkey.removesuffix(".k_proj.bias")
					bind=1
				else:
					bkey=mkey.removesuffix(".v_proj.bias")
					bind=2
				if not(bkey in bias_dict):
					bias_dict[bkey]=[None,None,None]
				bias_dict[bkey][bind]=p
				if bias_dict[bkey][0]!=None and bias_dict[bkey][1]!=None and bias_dict[bkey][2]!=None:
					teen2_sd["conditioner.embedders.1.model." +wkey+".in_proj_bias"]=torch.cat(bias_dict[bkey])
				continue

			teen2_sd["conditioner.embedders.1.model." +mkey]=p
		del teen2_sd_org

		if os.path.exists(vae_path):
			vae_sd_org=load_file(vae_path)
		else:
			vae_path = path+"/vae/diffusion_pytorch_model.bin"
			vae_sd_org=torch.load(vae_path)
		vae_sd={}
		for k,p in vae_sd_org.items():
			mkey=k
			for k2 in sdxl.vae_conversion_map:
				mkey=mkey.replace(sdxl.vae_conversion_map[k2],k2)
			if "attentions" in k:
				for k2 in sdxl.vae_conversion_map_attn:
					mkey = mkey.replace(sdxl.vae_conversion_map_attn[k2], k2)
			for k2 in ["q", "k", "v", "proj_out"]:
				if "mid.attn_1."+k2+".weight" in mkey:
					if p.ndim != 1:
						p=p.reshape(*p.shape,1,1)
			vae_sd["first_stage_model."+mkey]=p
		del vae_sd_org
		return {**unet_sd,**teen1_sd,**teen2_sd,**vae_sd}

	def lora(self,path):
		sd=load_file(path)

		out_sd={}
		f=open(path+".txt","w")
		for key,p in sd.items():
			if not(key.endswith((".lora_A.weight",".lora_down.weight"))):
				continue
			k=key.replace(".","_")
			if k.startswith("lora_unet"):
				if "resnets" in k:
					for k2 in sdxl.unet_conversion_map_resnet:
						k = k.replace(sdxl.unet_conversion_map_resnet[k2].replace(".","_"), k2.replace(".","_"))
				for k2 in sdxl.unet_conversion_map_layer:
					k = k.replace(k2[1].replace(".","_"), k2[0].replace(".","_"))
				if k.endswith("_lora_down_weight"):
					k=k.replace("_lora_down_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_down.weight",".lora_up.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_down.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				else:
					k=k.replace("_lora_A_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_A.weight",".lora_B.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_A.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				continue
			elif k.startswith("lora_te1"):
				if k.startswith("lora_te1_encoder"):
					k=k.replace("lora_te1_encoder","lora_te1_text_model_encoder")
				if k.endswith("_lora_down_weight"):
					k=k.replace("_lora_down_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_down.weight",".lora_up.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_down.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				else:
					k=k.replace("_lora_A_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_A.weight",".lora_B.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_A.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				continue
			elif k.startswith("lora_te2"):
				if k.startswith("lora_te2_encoder"):
					k=k.replace("lora_te2_encoder","lora_te2_text_model_encoder")
				if k.endswith("_lora_down_weight"):
					k=k.replace("_lora_down_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_down.weight",".lora_up.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_down.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				else:
					k=k.replace("_lora_A_weight",".lora_down.weight")
					out_sd[k]=p
					k=k.replace(".lora_down.weight",".lora_up.weight")
					k2= key.replace(".lora_A.weight",".lora_B.weight")
					out_sd[k]=sd[k2]
					k=k.replace(".lora_up.weight",".alpha")
					k2= key.replace(".lora_A.weight",".alpha")
					if k2 in sd:
						out_sd[k]=sd[k2]
				continue
			f.write(key+"\n")
		f.close()
		return out_sd

	def pipelora(self,path,unet=True,teen1=True,teen2=True):
		pipe = StableDiffusionXLPipeline.from_single_file(path)
		out_sd={}
		if unet:
			for name, module in pipe.unet.named_modules():
				if module.__class__.__name__ in ["Transformer2DModel"]:
					for child_name, child_module in module.named_modules():
						is_linear = child_module.__class__.__name__ == "Linear"
						is_conv2d = child_module.__class__.__name__ == "Conv2d"
						if is_linear or is_conv2d:
							lora_name = "lora_unet" + "." + name + "." + child_name
							lora_name = lora_name.replace(".", "_")
							for k2 in sdxl.unet_conversion_map_layer:
								lora_name = lora_name.replace(k2[1].replace(".","_"), k2[0].replace(".","_"))
							out_sd[lora_name]=child_module.weight.contiguous()
		if teen1:
			for name, module in pipe.text_encoder.named_modules():
				if module.__class__.__name__ in ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]:
					for child_name, child_module in module.named_modules():
						is_linear = child_module.__class__.__name__ == "Linear"
						is_conv2d = child_module.__class__.__name__ == "Conv2d"
						if is_linear or is_conv2d:
							lora_name = "lora_te1" + "." + name + "." + child_name
							lora_name = lora_name.replace(".", "_")
							out_sd[lora_name]=child_module.weight.contiguous()
		if teen2:
			for name, module in pipe.text_encoder_2.named_modules():
				if module.__class__.__name__ in ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]:
					for child_name, child_module in module.named_modules():
						is_linear = child_module.__class__.__name__ == "Linear"
						is_conv2d = child_module.__class__.__name__ == "Conv2d"
						if is_linear or is_conv2d:
							lora_name = "lora_te2" + "." + name + "." + child_name
							lora_name = lora_name.replace(".", "_")
							out_sd[lora_name]=child_module.weight.contiguous()
		del pipe
		return out_sd
