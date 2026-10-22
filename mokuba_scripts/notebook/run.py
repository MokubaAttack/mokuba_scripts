import json
from IPython.display import clear_output
import os
from accelerate.commands.launch import (
	launch_command_parser,
	launch_command
)

from ..common.yaml import mk_yaml

path=__file__.replace("\\","/")

def mokuani2(
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
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
	token="",
	gpu_num=1,
):
	params={
	"loras":loras,
	"lora_weights":lora_weights,
	"prompt":prompt,
	"n_prompt":n_prompt,
	"pic_number":pic_number,
	"gs":gs,
	"step":step,
	"sample":sample,
	"sgm":sgm,
	"seed":seed,
	"out_folder":out_folder,
	"base_safe":base_safe,
	"url":url,
	"dtype":dtype,
	"x":x,
	"y":y,
	"mode":mode,
	"up":up,
	"Interpolation":Interpolation,
	"step2":step2,
	"ss":ss,
	"token":token,
	}
	f=open("params.json","w")
	json.dump(params,f,indent=2)
	f.close()
	
	path1=path.replace("notebook/run.py","run/anima_run.py")
	path2=mk_yaml(gpu_num)
	parser = launch_command_parser()
	args = parser.parse_args(["--config_file",path2,path1])
	launch_command(args)

	if os.path.exists("params.json"):
		os.remove("params.json")
	if os.path.exists(path2):
		os.remove(path2)

def mokusdxl2(
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
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
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
	gpu_num=1,
):
	params={
	"loras":loras,
	"lora_weights":lora_weights,
	"prompt":prompt,
	"n_prompt":n_prompt,
	"pic_number":pic_number,
	"gs":gs,
	"step":step,
	"sample":sample,
	"sgm":sgm,
	"seed":seed,
	"out_folder":out_folder,
	"base_safe":base_safe,
	"url":url,
	"dtype":dtype,
	"x":x,
	"y":y,
	"mode":mode,
	"up":up,
	"Interpolation":Interpolation,
	"step2":step2,
	"ss":ss,
	"token":token,
	"pos_emb":pos_emb,
	"neg_emb":neg_emb,
	"vae_safe":vae_safe,
	"step3":step3,
	"up2":up2,
	"ccs":ccs,
	"gpulowmem":gpulowmem,
	"freezeunet":freezeunet,
	"cs":cs,
	"qprompt":qprompt,
	"qn_prompt":qn_prompt,
	}
	f=open("params.json","w")
	json.dump(params,f,indent=2)
	f.close()
	
	path2=mk_yaml(gpu_num)
	path1=path.replace("notebook/run.py","run/sdxl_run.py")
	parser = launch_command_parser()
	args = parser.parse_args(["--config_file",path2,path1])
	launch_command(args)

	if os.path.exists("params.json"):
		os.remove("params.json")
	if os.path.exists(path2):
		os.remove(path2)

def mokusd2(
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
	x=1024,
	y=1024,
	mode=0,
	up=1.5,
	Interpolation="BILINEAR",
	step2=15,
	ss=0.5,
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
	gpu_num=1,
):
	params={
	"loras":loras,
	"lora_weights":lora_weights,
	"prompt":prompt,
	"n_prompt":n_prompt,
	"pic_number":pic_number,
	"gs":gs,
	"step":step,
	"sample":sample,
	"sgm":sgm,
	"seed":seed,
	"out_folder":out_folder,
	"base_safe":base_safe,
	"url":url,
	"dtype":dtype,
	"x":x,
	"y":y,
	"mode":mode,
	"up":up,
	"Interpolation":Interpolation,
	"step2":step2,
	"ss":ss,
	"token":token,
	"pos_emb":pos_emb,
	"neg_emb":neg_emb,
	"vae_safe":vae_safe,
	"step3":step3,
	"up2":up2,
	"ccs":ccs,
	"gpulowmem":gpulowmem,
	"freezeunet":freezeunet,
	"cs":cs,
	"qprompt":qprompt,
	"qn_prompt":qn_prompt,
	}
	f=open("params.json","w")
	json.dump(params,f,indent=2)
	f.close()
	
	path2=mk_yaml(gpu_num)
	path1=path.replace("notebook/run.py","run/sd_run.py")
	parser = launch_command_parser()
	args = parser.parse_args(["--config_file",path2,path1])
	launch_command(args)

	if os.path.exists("params.json"):
		os.remove("params.json")
	if os.path.exists(path2):
		os.remove(path2)
