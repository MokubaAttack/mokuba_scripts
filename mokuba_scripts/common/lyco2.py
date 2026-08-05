import re
import torch
from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.full import FullModule
from lycoris.modules.norms import NormModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.dylora import DyLoraModule
from lycoris.modules.ia3 import IA3Module
from safetensors.torch import load_file

from ..common.keys import (
	anima,
	sdxl,
	sdkeys
)

MODULE_LIST = [
	LoConModule,
	LohaModule,
	IA3Module,
	LokrModule,
	FullModule,
	NormModule,
	DiagOFTModule,
	ButterflyOFTModule,
	GLoRAModule,
	DyLoraModule,
]

def lyco2anima(path):
	root_map=anima.root_map
	block_maps=anima.block_maps
	sd=load_file(path)

	MODULE_type=None
	for m in MODULE_LIST:
		for k in m.weight_list_det:
			for k2 in sd:
				if k2.endswith(("lora_B.weight","lora_up.weight")):
					MODULE_type="B"
					break
				if k2.endswith(k):
					MODULE_type=m
					break
			if MODULE_type!=None:
				break
		if MODULE_type!=None:
			break
	if MODULE_type==None:
		raise RuntimeWarning(path+" isn't supported.")
	if MODULE_type=="B":
		MODULE_type=LoConModule
		key_dict=list(sd)
		for k2 in key_dict:
			if k2.endswith("lora_B.weight"):
				k=k2.replace("lora_B.weight","lora_up.weight")
				sd[k]=sd.pop(k2)
				kk=k2.replace("lora_B.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k].size()[1])
			elif k2.endswith("lora_up.weight"):
				kk=k2.replace("lora_up.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k2].size()[1])
			elif k2.endswith("lora_A.weight"):
				k=k2.replace("lora_A.weight","lora_down.weight")
				sd[k]=sd.pop(k2)
	key_dict={}
	for k in sd:
		for k2 in MODULE_type.weight_list_det:
			if k.endswith("."+k2):
				k=k.removesuffix("."+k2)
				key_dict[k]=k.replace(".","_")

	transformer_sd={}
	text_encoder_sd={}
	text_conditioner_sd={}

	for k in key_dict:
		m=re.search(r"layers_([0-9]+)_(.+)$",key_dict[k])
		if m is not None:
			key_dict[k]=m.group()
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					text_encoder_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
			continue

		m=re.search(r"llm_adapter_blocks",key_dict[k])
		if m!=None:
			key_dict[k]="blocks"+key_dict[k][m.end():]
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					text_conditioner_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
			continue

		m=re.search(r"blocks_[0-9]+_",key_dict[k])
		if m!=None:
			key_dict[k]=key_dict[k][m.start():]
			for k2 in block_maps:
				mk2=k2.removesuffix(".weight").replace(".","_")
				mk2_value=block_maps[k2].removesuffix(".weight").replace(".","_")
				key_dict[k]=key_dict[k].replace(mk2,mk2_value)
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					transformer_sd["lycoris_transformer_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
			continue

		for k3 in root_map:
			mk2=k3.removesuffix(".weight").replace(".","_")
			mk2_value=root_map[k3].removesuffix(".weight").replace(".","_")
			m=re.search(mk2,key_dict[k])
			if m!=None:
				key_dict[k]=mk2_value
				for k2 in MODULE_type.weight_list:
					if k+"."+k2 in sd:
						transformer_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
				break
			m=re.search(mk2_value,key_dict[k])
			if m!=None:
				key_dict[k]=mk2_value
				for k2 in MODULE_type.weight_list:
					if k+"."+k2 in sd:
						transformer_sd["lycoris_"+key_dict[k]+"."+k2]=sd.pop(k+"."+k2)
				break
	return transformer_sd,text_conditioner_sd,text_encoder_sd

def lyco2sdxl(path):
	sd=load_file(path)

	MODULE_type=None
	for m in MODULE_LIST:
		for k in m.weight_list_det:
			for k2 in sd:
				if k2.endswith(("lora_B.weight","lora_up.weight")):
					MODULE_type="B"
					break
				if k2.endswith(k):
					MODULE_type=m
					break
			if MODULE_type!=None:
				break
		if MODULE_type!=None:
			break
	if MODULE_type==None:
		raise RuntimeWarning(path+" isn't supported.")
	if MODULE_type=="B":
		MODULE_type=LoConModule
		key_dict=list(sd)
		for k2 in key_dict:
			if k2.endswith("lora_B.weight"):
				k=k2.replace("lora_B.weight","lora_up.weight")
				sd[k]=sd.pop(k2)
				kk=k2.replace("lora_B.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k].size()[1])
			elif k2.endswith("lora_up.weight"):
				kk=k2.replace("lora_up.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k2].size()[1])
			elif k2.endswith("lora_A.weight"):
				k=k2.replace("lora_A.weight","lora_down.weight")
				sd[k]=sd.pop(k2)
	key_dict={}
	key_dict_swap={}
	for k in sd:
		for k2 in MODULE_type.weight_list_det:
			if k.endswith("."+k2):
				k=k.removesuffix("."+k2)
				key_dict[k]=k.replace(".","_")
				key_dict_swap[k.replace(".","_")]=k

	unet_sd={}
	text_encoder_sd={}
	text_encoder_2_sd={}

	for k,w in sdxl.unet_conversion_map.items():
		m="lora_unet_"+k.removesuffix(".").replace(".","_")
		if m in key_dict_swap:
			del key_dict[key_dict_swap[m]]
			for k2 in MODULE_type.weight_list:
				if key_dict_swap[m]+"."+k2 in sd:
					unet_sd["lycoris_"+w.removesuffix(".").replace(".","_")+"."+k2]=sd.pop(key_dict_swap[m]+"."+k2)

	for k,m in key_dict.items():
		if m.startswith("lora_unet_"):
			m=m.removeprefix("lora_unet_")
			m=m.replace("output_blocks_2_2_conv","up_blocks_0_upsamplers_0_conv")
			for k2 in sdxl.unet_conversion_map_layer:
				m=m.replace(k2[0].removesuffix(".").replace(".","_"),k2[1].removesuffix(".").replace(".","_"))
			if "resnets" in m:
				for k2 in sdxl.unet_conversion_map_resnet:
					m=m.replace(k2.replace(".","_"),sdxl.unet_conversion_map_resnet[k2])
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					unet_sd["lycoris_"+m+"."+k2]=sd.pop(k+"."+k2)
		elif m.startswith("lora_te1_"):
			m=m.removeprefix("lora_te1_text_model_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					text_encoder_sd["lycoris_"+m+"."+k2]=sd.pop(k+"."+k2)
		elif m.startswith("lora_te2_"):
			m=m.removeprefix("lora_te2_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					text_encoder_2_sd["lycoris_"+m+"."+k2]=sd.pop(k+"."+k2)

	return unet_sd,text_encoder_sd,text_encoder_2_sd
	
def lyco2sd(path):
	sd=load_file(path)

	MODULE_type=None
	for m in MODULE_LIST:
		for k in m.weight_list_det:
			for k2 in sd:
				if k2.endswith(("lora_B.weight","lora_up.weight")):
					MODULE_type="B"
					break
				if k2.endswith(k):
					MODULE_type=m
					break
			if MODULE_type!=None:
				break
		if MODULE_type!=None:
			break
	if MODULE_type==None:
		raise RuntimeWarning(path+" isn't supported.")
	if MODULE_type=="B":
		MODULE_type=LoConModule
		key_dict=list(sd)
		for k2 in key_dict:
			if k2.endswith("lora_B.weight"):
				k=k2.replace("lora_B.weight","lora_up.weight")
				sd[k]=sd.pop(k2)
				kk=k2.replace("lora_B.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k].size()[1])
			elif k2.endswith("lora_up.weight"):
				kk=k2.replace("lora_up.weight","alpha")
				if not(kk in sd):
					sd[kk]=torch.tensor(sd[k2].size()[1])
			elif k2.endswith("lora_A.weight"):
				k=k2.replace("lora_A.weight","lora_down.weight")
				sd[k]=sd.pop(k2)
	key_dict={}
	key_dict_swap={}
	for k in sd:
		for k2 in MODULE_type.weight_list_det:
			if k.endswith("."+k2):
				k=k.removesuffix("."+k2)
				key_dict[k]=k.replace(".","_")
				key_dict_swap[k.replace(".","_")]=k

	unet_sd={}
	text_encoder_sd={}

	for k,w in sdkeys.unet_conversion_map.items():
		m="lora_unet_"+k.removesuffix(".").replace(".","_")
		if m in key_dict_swap:
			del key_dict[key_dict_swap[m]]
			for k2 in MODULE_type.weight_list:
				if key_dict_swap[m]+"."+k2 in sd:
					unet_sd["lycoris_"+w.removesuffix(".").replace(".","_")+"."+k2]=sd.pop(key_dict_swap[m]+"."+k2)

	for k,m in key_dict.items():
		if m.startswith("lora_unet_"):
			m=m.removeprefix("lora_unet_")
			for k2 in sdkeys.unet_conversion_map_layer:
				m=m.replace(k2[0].removesuffix(".").replace(".","_"),k2[1].removesuffix(".").replace(".","_"))
			if "resnets" in m:
				for k2 in sdkeys.unet_conversion_map_resnet:
					m=m.replace(k2.replace(".","_"),sdkeys.unet_conversion_map_resnet[k2])
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					unet_sd["lycoris_"+m+"."+k2]=sd.pop(k+"."+k2)
		elif m.startswith("lora_te_"):
			m=m.removeprefix("lora_te_text_model_")
			for k2 in MODULE_type.weight_list:
				if k+"."+k2 in sd:
					text_encoder_sd["lycoris_"+m+"."+k2]=sd.pop(k+"."+k2)
	return unet_sd,text_encoder_sd
