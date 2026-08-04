import os
import json
from huggingface_hub import snapshot_download
import re
from safetensors.torch import load_file

from ..common.keys import anima

root_map=anima.root_map
block_maps=anima.block_maps
vae_keys1=anima.vae_keys1
vae_keys2=anima.vae_keys2
vae_keys3=anima.vae_keys3

def check_anima_base():
	if not(os.path.exists(os.getcwd()+"/AnimaBaseV1")):
		snapshot_download(repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers", local_dir=os.getcwd()+"/AnimaBaseV1")

		json_path=os.getcwd()+"/AnimaBaseV1/modular_model_index.json"
		f=open(json_path,"r")
		json_sd=json.load(f)
		f.close()

		json_sd["scheduler"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["t5_tokenizer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["transformer"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"
		json_sd["vae"][-1]["pretrained_model_name_or_path"]=os.getcwd()+"/AnimaBaseV1"

		f=open(json_path,"w")
		json.dump(json_sd, f, indent=2)
		f.close()

		f=open(os.getcwd()+"/AnimaBaseV1/id.txt","w")
		f.write("2945208")
		f.close()

def safe2diff(path,trans=True,teco=True,teen=True,vae=True):
	check_anima_base()

	sd=load_file(path)
	keys=[]
	head_trans=None
	head_teen=None
	head_teco=None
	head_vae=None
	check_trans=("final_layer.linear.weight","proj_out.weight")
	check_teco="llm_adapter"
	check_teen="embed_tokens.weight"
	check_vae="decoder"

	for k in sd:
		keys.append(k)
		if k.endswith(check_trans):
			head_trans=k.removesuffix(check_trans[0])
			head_trans=head_trans.removesuffix(check_trans[1])
		elif k.endswith(check_teen):
			head_teen=k.removesuffix(check_teen)
		m=re.search(check_teco+r"\.(.+)",k)
		if m is not None:
			head_teco=k.removesuffix(m.group(1))
		m=re.search(check_vae+r"(.+)",k)
		if m is not None:
			head_vae=k.removesuffix(check_vae+m.group(1))

	text_conditioner_sd={}
	transformer_sd={}
	vae_sd={}
	text_encoder_sd={}

	plus_key=[[],[],[],[]]
	minus_key=[[],[],[],[],[]]

	for k in keys:
		if head_teco!=None and teco:
			if k.startswith(head_teco):
				mk=k.removeprefix(head_teco)
				text_conditioner_sd[mk]=sd.pop(k)
				continue

		if head_vae!=None and vae:
			if k.startswith(head_vae):
				mk=k.removeprefix(head_vae)
				if mk.startswith("encoder."):
					for k2 in vae_keys2:
						mk=mk.replace(k2,vae_keys2[k2])
				elif mk.startswith("decoder."):
					for k2 in vae_keys3:
						mk=mk.replace(k2,vae_keys3[k2])
				else:
					for k2 in vae_keys1:
						mk=mk.replace(k2,vae_keys1[k2])
				vae_sd[mk]=sd.pop(k)
				continue

		if head_teen!=None and teen:
			if k.startswith(head_teen):
				mk=k.removeprefix(head_teen)
				text_encoder_sd[mk]=sd.pop(k)
				continue

		if head_trans!=None and trans:
			if k.startswith(head_trans):
				mk=k.removeprefix(head_trans)
				m = root_map.get(mk)
				if m is not None:
					transformer_sd[m] = sd.pop(k)
					continue
				elif mk in root_map.values():
					transformer_sd[mk] = sd.pop(k)
					continue

				m = re.match(r"blocks\.(\d+)\.(.+)",mk)
				if m is not None:
					block_index = m.group(1)
					tail = m.group(2)
					mapped_tail = block_maps.get(tail)
					if tail in block_maps.values():
						mapped_tail=tail
					if mapped_tail is not None:
						transformer_sd[f"transformer_blocks.{block_index}.{mapped_tail}"] = sd.pop(k)
						continue
		minus_key.append("unknown."+k)

	if transformer_sd=={} and text_encoder_sd=={} and text_conditioner_sd=={} and vae_sd=={}:
		raise RuntimeWarning(path+" isn't supported.")

	if teco:
		sd2=load_file(os.getcwd()+"/AnimaBaseV1/text_conditioner/diffusion_pytorch_model.safetensors")
		for k in sd2:
			if not(k in text_conditioner_sd):
				text_conditioner_sd[k]=sd2[k]
				plus_key[1].append("text_conditioner."+k)
		keys=list(text_conditioner_sd)
		for k in keys:
			if not(k in sd2):
				del text_conditioner_sd[k]
				minus_key[1].append("text_conditioner."+k)
		del sd2

	if trans:
		sd2=load_file(os.getcwd()+"/AnimaBaseV1/transformer/diffusion_pytorch_model.safetensors")
		for k in sd2:
			if not(k in transformer_sd):
				transformer_sd[k]=sd2[k]
				plus_key[0].append("transformer."+k)
		keys=list(transformer_sd)
		for k in keys:
			if not(k in sd2):
				del transformer_sd[k]
				minus_key[0].append("transformer."+k)
		del sd2

	if vae:
		sd2=load_file(os.getcwd()+"/AnimaBaseV1/vae/diffusion_pytorch_model.safetensors")
		for k in sd2:
			if not(k in vae_sd):
				vae_sd[k]=sd2[k]
				plus_key[3].append("vae."+k)
		keys=list(vae_sd)
		for k in keys:
			if not(k in sd2):
				del vae_sd[k]
				minus_key[3].append("vae."+k)
		del sd2

	if teen:
		sd2=load_file(os.getcwd()+"/AnimaBaseV1/text_encoder/model.safetensors")
		for k in sd2:
			if not(k in text_encoder_sd):
				text_encoder_sd[k]=sd2[k]
				plus_key[2].append("text_encoder."+k)
		keys=list(text_encoder_sd)
		for k in keys:
			if not(k in sd2):
				del text_encoder_sd[k]
				minus_key[2].append("text_encoder."+k)
		del sd2
	
	f=open(path+".txt","w")
	f.write("minus\n")
	for ks in minus_key:
		for k in ks:
			f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()
	
	return (transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd)

def folder2diff(path,trans=True,teco=True,teen=True,vae=True):
	check_anima_base()
	if not(os.path.exists(path+"/modular_model_index.json")):
		raise RuntimeWarning("modular_model_index.json does not exist.")
	f=open(path+"/modular_model_index.json","r")
	json_sd=json.load(f)
	f.close()

	text_conditioner_path=json_sd["text_conditioner"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["text_conditioner"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	text_encoder_path=json_sd["text_encoder"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["text_encoder"][-1]["subfolder"]+"/model.safetensors"
	transformer_path=json_sd["transformer"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["transformer"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	vae_path=json_sd["vae"][-1]["pretrained_model_name_or_path"]+"/"+json_sd["vae"][-1]["subfolder"]+"/diffusion_pytorch_model.safetensors"
	
	plus_key=[[],[],[],[]]
	minus_key=[[],[],[],[]]
	
	if trans:
		if not(os.path.exists(transformer_path)):
			raise RuntimeWarning("transformer file does not exist.")
		sd1=load_file(transformer_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/transformer/diffusion_pytorch_model.safetensors")
		for k in sd22:
			if not(k in sd1):
				sd1[k]=sd22[k]
				plus_key[0].append("transformer."+k)
		keys=list(sd1)
		for k in keys:
			if not(k in sd22):
				del sd1[k]
				minus_key[0].append("transformer."+k)
	else:
		sd1={}

	if teco:	
		if not(os.path.exists(text_conditioner_path)):
			raise RuntimeWarning("text_conditioner file does not exist.")
		sd2=load_file(text_conditioner_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/text_conditioner/diffusion_pytorch_model.safetensors")
		for k in sd22:
			if not(k in sd2):
				sd2[k]=sd22[k]
				plus_key[1].append("text_conditioner."+k)
		keys=list(sd2)
		for k in keys:
			if not(k in sd22):
				del sd2[k]
				minus_key[1].append("text_conditioner."+k)
	else:
		sd2={}

	if teen:
		if not(os.path.exists(text_encoder_path)):
			raise RuntimeWarning("text_encoder file does not exist.")
		sd3=load_file(text_encoder_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/text_encoder/model.safetensors")
		for k in sd22:
			if not(k in sd3):
				sd3[k]=sd22[k]
				plus_key[2].append("text_encoder."+k)
		keys=list(sd3)
		for k in keys:
			if not(k in sd22):
				del sd3[k]
				minus_key[2].append("text_encoder."+k)
	else:
		sd3={}

	if vae:	
		if not(os.path.exists(vae_path)):
			raise RuntimeWarning("vae file does not exist.")
		sd4=load_file(vae_path)
		sd22=load_file(os.getcwd()+"/AnimaBaseV1/vae/diffusion_pytorch_model.safetensors")
		for k in sd22:
			if not(k in sd4):
				sd4[k]=sd22[k]
				plus_key[3].append("vae."+k)
		keys=list(sd4)
		for k in keys:
			if not(k in sd22):
				del sd4[k]
				minus_key[3].append("vae."+k)
	else:
		sd4={}
		
	f=open(path+".txt","w")
	f.write("minus\n")
	for ks in minus_key:
		for k in ks:
			f.write(k+"\n")
	f.write("plus\n")
	for ks in plus_key:
		for k in ks:
			f.write(k+"\n")
	f.close()
	
	return (sd1,sd2,sd3,sd4)