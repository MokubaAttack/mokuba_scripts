from mokuba_scripts.notebook.workflow import mokusd
import json
import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"

if __name__ == "__main__":
    f=open("pipecache/params.json","r")
    sd=json.load(f)
    f.close()
    
	pipe=mokusd(
		base_safe=sd["base_safe"],
		loras=sd["loras"],
		lora_weights=sd["lora_weights"],
		prompt=sd["prompt"],
		n_prompt=sd["n_prompt"],
		step=sd["step"],
		gs=sd["gs"],
		sample=sd["sample"],
		sgm=sd["sgm"],
		pic_number=sd["pic_number"],
		seed=sd["seed"],
		out_folder=sd["out_folder"],
		url=sd["url"],
		x=sd["x"],
		y=sd["y"],
		dtype=sd["dtype"],
		up=sd["up"],
		mode=sd["mode"],
		Interpolation=sd["Interpolation"],
		step2=sd["step2"],
		ss=sd["ss"],
		token=sd["token"],
		pos_emb=sd["pos_emb"],
		neg_emb=sd["neg_emb"],
		vae_safe=sd["vae_safe"],
		step3=sd["step3"],
		up2=sd["up2"],
		ccs=sd["ccs"],
		gpulowmem=sd["gpulowmem"],
		freezeunet=sd["freezeunet"],
		cs=sd["cs"],
		qprompt=sd["qprompt"],
		qn_prompt=sd["qn_prompt"],
    )
