import mokuba_scripts
import json
import os
import shutil

f=open("params.json","r")
sd=json.load(f)
f.close()
if os.path.exists(sd["out_folder"]):
    shutil.rmtree(sd["out_folder"])
pipe=mokuba_scripts.mokuani(
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
    si=sd["si"],
    url=sd["url"],
    del_pipe=sd["del_pipe"],
    x=sd["x"],
    y=sd["y"],
    p=sd["pipe"],
    dtype=sd["dtype"],
    dev=sd["dev"],
    up=sd["up"],
    mode=sd["mode"],
    Interpolation=sd["Interpolation"],
    step2=sd["step2"],
    ss=sd["ss"],
    token=sd["token"],
)
