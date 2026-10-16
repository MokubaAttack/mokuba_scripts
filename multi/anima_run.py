import mokuba_scripts
import json

f=open("params.json","r")
sd=json.load(f)
f.close()

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
)
