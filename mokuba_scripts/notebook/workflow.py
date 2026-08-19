import torch
import os
import math
from diffusers.utils import make_image_grid
from PIL import Image
from IPython.display import (
	display,
	clear_output
)
import time
import shutil
import re

from ..anima.mokuanipipe import mokuanipipe
from ..sd.mokusdxlpipe import mokusdxlpipe
from ..sd.mokusdpipe import mokusdpipe
from ..common.discord import up_drop
from ..common.flush import (
	flush,
	reset_func
)
from ..common.metadata import plus_meta
from ..common.seed import make_seed
from ..common.dl import dlc

def imgshow(imgs):
	r=math.ceil(len(imgs)/2)
	if r*2!=len(imgs):
		simgs=imgs+[Image.new('RGB', imgs[-1].size, (0, 0, 0))]
	else:
		simgs=imgs
	x,y=imgs[-1].size
	x=round(2*x/y*400)
	display(make_image_grid(simgs, rows=r, cols=2).resize((x,400*r)))

def mokuani(
	loras=[],
	lora_weights=[],
	prompt = "",
	n_prompt = "",
	pic_number=10,
	gs=7,
	step=30,
	sample="",
	sgm="",
	seed=0,
	out_folder="data",
	base_safe="base.safetensors",
	url="",
	dtype="f32",
	dev="cuda",
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
	p=None,
	ser="colab",
	del_pipe=True,
	si=True,
	token="",
):
	ut=round(time.time())
	if not(isinstance(url, list)):
		url=[]
	if len(url)!=3:
		url=[]
	seed,pic_number=make_seed(seed,pic_number)

	if p==None:
		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="f16":
			dtype=torch.float16
		else:
			dtype=torch.float32

		base_safe=str(base_safe)
		m=re.match(r"[0-9]+$",base_safe)
		if m!=None:
			ver_id=base_safe
			base_safe=base_safe+".safetensors"
			if token=="":
				raise RuntimeWarning("civitai token doesn't input.")
			dlc(ver_id,base_safe,token)

		pipe=mokuanipipe()
		if base_safe.endswith(".safetensors"):
			pipe.from_safe(path=base_safe,torch_dtype=dtype,device=dev)
		else:
			pipe.from_folder(path=base_safe,torch_dtype=dtype,device=dev)

		pipe.set_scheduler(sample,sgm)

		if len(loras)!=len(lora_weights):
			raise RuntimeWarning("the number of lora does not equal the number of lora weight.")
		for line,w in zip(loras,lora_weights):
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_lycoris(path=line,weight=w)
	else:
		pipe=p

	if not(os.path.exists(out_folder+"/0")):
		os.makedirs(out_folder+"/0")

	imgs=[]
	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.txt2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step,
			x=x,
			y=y,
			seed=s
		)
		imgs.append(img)

		meta["input"]=out_folder+"/0/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/0/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()
		
	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)

	if mode==0:
		if url!=[]:
			shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
			drop_path=str(ut)+".zip"
			up_drop("archive_shutil.zip",drop_path,url)
			os.remove("archive_shutil.zip")
		if del_pipe:
			reset_func(pipe,ser)
			pipe=None
		return pipe

	if not(os.path.exists(out_folder+"/1")):
		os.makedirs(out_folder+"/1")

	pipe.make_upscaler(Interpolation,dev)
	x=round(x*up/8)*8
	y=round(y*up/8)*8
	for i,img in enumerate(imgs):
		imgs[i]=pipe.upscale(img,x,y)
	pipe.delete_upscaler()

	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.img2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step2,
			img=imgs[i],
			seed=s,
			ss=ss,
			hiresfix=True
		)
		imgs[i]=img

		meta["input"]=out_folder+"/1/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/1/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()
		
	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if url!=[]:
		shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
		drop_path=str(ut)+".zip"
		up_drop("archive_shutil.zip",drop_path,url)
		os.remove("archive_shutil.zip")
	if del_pipe:
		reset_func(pipe,ser)
		pipe=None
	return pipe
	
def mokusdxl(
	loras=[],
	lora_weights=[],
	prompt = "",
	n_prompt = "",
	pic_number=10,
	gs=7,
	step=30,
	sample="",
	sgm="",
	seed=0,
	out_folder="data",
	base_safe="base.safetensors",
	url="",
	dtype="f32",
	dev="cuda",
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
	p=None,
	ser="colab",
	del_pipe=True,
	si=True,
	pos_emb=[],
	neg_emb=[],
	vae_safe="",
	step3=20,
	up2=1.5,
	ccs=0,
	gpulowmem=False,
	freezeunet=False,
	cs=2,
	qprompt="masterpiece, best quality, ultra detailed",
	qn_prompt="worst quality, low quality, normal quality",
	token="",
):
	ut=round(time.time())
	if not(isinstance(url, list)):
		url=[]
	if len(url)!=3:
		url=[]
	seed,pic_number=make_seed(seed,pic_number)

	if p==None:
		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="f16":
			dtype=torch.float16
		else:
			dtype=torch.float32

		base_safe=str(base_safe)
		m=re.match(r"[0-9]+$",base_safe)
		if m!=None:
			ver_id=base_safe
			base_safe=base_safe+".safetensors"
			if token=="":
				raise RuntimeWarning("civitai token doesn't input.")
			dlc(ver_id,base_safe,token)

		vae_safe=str(vae_safe)
		m=re.match(r"[0-9]+$",vae_safe)
		if m!=None:
			ver_id=vae_safe
			vae_safe=vae_safe+".safetensors"
			if token=="":
				raise RuntimeWarning("civitai token doesn't input.")
			dlc(ver_id,vae_safe,token)

		pipe=mokusdxlpipe()
		if base_safe.endswith(".safetensors"):
			pipe.from_safe(path=base_safe,torch_dtype=dtype,device=dev,vae_path=vae_safe)
		else:
			pipe.from_folder(path=base_safe,torch_dtype=dtype,device=dev,vae_path=vae_safe)

		pipe.set_scheduler(sample,sgm)

		if len(loras)!=len(lora_weights):
			raise RuntimeWarning("the number of lora does not equal the number of lora weight.")
		for line,w in zip(loras,lora_weights):
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_lycoris(path=line,weight=w)

		for line in pos_emb:
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_pos_embed(line)

		for line in neg_emb:
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_neg_embed(line)

		if gpulowmem:
			pipe.set_gpu_lowmem()
		if freezeunet:
			pipe.set_freeze_unet()
	else:
		pipe=p

	if not(os.path.exists(out_folder+"/0")):
		os.makedirs(out_folder+"/0")

	imgs=[]
	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.txt2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step,
			x=x,
			y=y,
			seed=s,
			cs=cs
		)
		imgs.append(img)

		meta["input"]=out_folder+"/0/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/0/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()

	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if mode==0:
		if url!=[]:
			shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
			drop_path=str(ut)+".zip"
			up_drop("archive_shutil.zip",drop_path,url)
			os.remove("archive_shutil.zip")
		if del_pipe:
			reset_func(pipe,ser)
			pipe=None
		return pipe

	if not(os.path.exists(out_folder+"/1")):
		os.makedirs(out_folder+"/1")

	pipe.make_upscaler(Interpolation,dev)
	x=round(x*up/8)*8
	y=round(y*up/8)*8
	for i,img in enumerate(imgs):
		imgs[i]=pipe.upscale(img,x,y)
	pipe.delete_upscaler()

	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.img2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step2,
			img=imgs[i],
			seed=s,
			ss=ss,
			cs=cs,
			hiresfix=True
		)
		imgs[i]=img

		meta["input"]=out_folder+"/1/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/1/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()

	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if mode==1:
		if url!=[]:
			shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
			drop_path=str(ut)+".zip"
			up_drop("archive_shutil.zip",drop_path,url)
			os.remove("archive_shutil.zip")
		if del_pipe:
			reset_func(pipe,ser)
			pipe=None
		return pipe

	if not(os.path.exists(out_folder+"/2")):
		os.makedirs(out_folder+"/2")

	pipe.make_upscaler(Interpolation,dev)
	if ccs==0:
		x=round(x*up2/8)*8
		y=round(y*up2/8)*8
	else:
		x=round(x*up2/64)*64
		y=round(y*up2/64)*64
	for i,img in enumerate(imgs):
		imgs[i]=pipe.upscale(img,x,y)
	pipe.delete_upscaler()

	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.tileup(
			prompt=qprompt,
			n_prompt=qn_prompt,
			gs=gs,
			step=step3,
			img=imgs[i],
			seed=s,
			ss=ss,
			ccs=ccs,
			cs=cs
		)
		imgs[i]=img

		meta["input"]=out_folder+"/2/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/2/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()
	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if url!=[]:
		shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
		drop_path=str(ut)+".zip"
		up_drop("archive_shutil.zip",drop_path,url)
		os.remove("archive_shutil.zip")
	if del_pipe:
		reset_func(pipe,ser)
		pipe=None
	return pipe
	
def mokusd(
	loras=[],
	lora_weights=[],
	prompt = "",
	n_prompt = "",
	pic_number=10,
	gs=7,
	step=30,
	sample="",
	sgm="",
	seed=0,
	out_folder="data",
	base_safe="base.safetensors",
	url="",
	dtype="f32",
	dev="cuda",
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
	p=None,
	ser="colab",
	del_pipe=True,
	si=True,
	pos_emb=[],
	neg_emb=[],
	vae_safe="",
	step3=20,
	up2=1.5,
	ccs=0,
	gpulowmem=False,
	freezeunet=False,
	cs=2,
	qprompt="masterpiece, best quality, ultra detailed",
	qn_prompt="worst quality, low quality, normal quality",
	token="",
):
	ut=round(time.time())
	if not(isinstance(url, list)):
		url=[]
	if len(url)!=3:
		url=[]
	seed,pic_number=make_seed(seed,pic_number)

	if p==None:
		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="f16":
			dtype=torch.float16
		else:
			dtype=torch.float32

		base_safe=str(base_safe)
		m=re.match(r"[0-9]+$",base_safe)
		if m!=None:
			ver_id=base_safe
			base_safe=base_safe+".safetensors"
			if token=="":
				raise RuntimeWarning("civitai token doesn't input.")
			dlc(ver_id,base_safe,token)

		vae_safe=str(vae_safe)
		m=re.match(r"[0-9]+$",vae_safe)
		if m!=None:
			ver_id=vae_safe
			vae_safe=vae_safe+".safetensors"
			if token=="":
				raise RuntimeWarning("civitai token doesn't input.")
			dlc(ver_id,vae_safe,token)

		pipe=mokusdpipe()
		if base_safe.endswith(".safetensors"):
			pipe.from_safe(path=base_safe,torch_dtype=dtype,device=dev,vae_path=vae_safe)
		else:
			pipe.from_folder(path=base_safe,torch_dtype=dtype,device=dev,vae_path=vae_safe)

		pipe.set_scheduler(sample,sgm)

		if len(loras)!=len(lora_weights):
			raise RuntimeWarning("the number of lora does not equal the number of lora weight.")
		for line,w in zip(loras,lora_weights):
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_lycoris(path=line,weight=w)

		for line in pos_emb:
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_pos_embed(line)

		for line in neg_emb:
			line=str(line)
			m=re.match(r"[0-9]+$",line)
			if m!=None:
				ver_id=line
				line=line+".safetensors"
				if token=="":
					raise RuntimeWarning("civitai token doesn't input.")
				dlc(ver_id,line,token)
			if not(line.endswith(".safetensors")):
				line=line+".safetensors"
			pipe.load_neg_embed(line)

		if gpulowmem:
			pipe.set_gpu_lowmem()
		if freezeunet:
			pipe.set_freeze_unet()
	else:
		pipe=p

	if not(os.path.exists(out_folder+"/0")):
		os.makedirs(out_folder+"/0")

	imgs=[]
	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.txt2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step,
			x=x,
			y=y,
			seed=s,
			cs=cs
		)
		imgs.append(img)

		meta["input"]=out_folder+"/0/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/0/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()

	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if mode==0:
		if url!=[]:
			shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
			drop_path=str(ut)+".zip"
			up_drop("archive_shutil.zip",drop_path,url)
			os.remove("archive_shutil.zip")
		if del_pipe:
			reset_func(pipe,ser)
			pipe=None
		return pipe

	if not(os.path.exists(out_folder+"/1")):
		os.makedirs(out_folder+"/1")

	pipe.make_upscaler(Interpolation,dev)
	x=round(x*up/8)*8
	y=round(y*up/8)*8
	for i,img in enumerate(imgs):
		imgs[i]=pipe.upscale(img,x,y)
	pipe.delete_upscaler()

	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.img2img(
			prompt=prompt,
			n_prompt=n_prompt,
			gs=gs,
			step=step2,
			img=imgs[i],
			seed=s,
			ss=ss,
			cs=cs,
			hiresfix=True
		)
		imgs[i]=img

		meta["input"]=out_folder+"/1/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/1/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()

	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if mode==1:
		if url!=[]:
			shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
			drop_path=str(ut)+".zip"
			up_drop("archive_shutil.zip",drop_path,url)
			os.remove("archive_shutil.zip")
		if del_pipe:
			reset_func(pipe,ser)
			pipe=None
		return pipe

	if not(os.path.exists(out_folder+"/2")):
		os.makedirs(out_folder+"/2")

	pipe.make_upscaler(Interpolation,dev)
	if ccs==0:
		x=round(x*up2/8)*8
		y=round(y*up2/8)*8
	else:
		x=round(x*up2/64)*64
		y=round(y*up2/64)*64
	for i,img in enumerate(imgs):
		imgs[i]=pipe.upscale(img,x,y)
	pipe.delete_upscaler()

	for i,s in enumerate(seed):
		clear_output(True)
		print(str(i)+"_"+str(s))
		if si and len(imgs)>0:
			imgshow(imgs)
		img,meta=pipe.tileup(
			prompt=qprompt,
			n_prompt=qn_prompt,
			gs=gs,
			step=step3,
			img=imgs[i],
			seed=s,
			ss=ss,
			ccs=ccs,
			cs=cs
		)
		imgs.append(img)

		meta["input"]=out_folder+"/2/"+str(i)+"_"+str(s)+".jpg"
		plus_meta(meta,img)
		if url!=[]:
			drop_path=str(ut)+"/2/"+str(i)+"_"+str(s)+".jpg"
			up_drop(meta["input"],drop_path,url)
		flush()
	clear_output(True)
	if si and len(imgs)>0:
		imgshow(imgs)
	if url!=[]:
		shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
		drop_path=str(ut)+".zip"
		up_drop("archive_shutil.zip",drop_path,url)
		os.remove("archive_shutil.zip")
	if del_pipe:
		reset_func(pipe,ser)
		pipe=None
	return pipe