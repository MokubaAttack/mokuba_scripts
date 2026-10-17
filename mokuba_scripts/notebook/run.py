import json
from IPython.display import clear_output
import subprocess

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
	
	path=__file__.replace("\\","/")
	path=path.replace("notebook/run.py","run/anima_run.py")
	output=subprocess.call("accelerate launch --num_processes=2 "+path, shell=True)
	
	if output==0:
		from IPython.display import clear_output
		clear_output(True)
		print("fin")
