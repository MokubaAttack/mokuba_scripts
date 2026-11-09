import safetensors
from PIL import PngImagePlugin
import piexif
import piexif.helper
import os
import ast

def plus_meta(vs,img):
	if "pr" in vs:
		if vs["pr"]=="":
			metadata="None\n"
		else:
			metadata=vs["pr"]+"\n"
	if "ne" in vs:
		if vs["ne"]=="":
			metadata=metadata+"Negative prompt: None\n"
		else:
			metadata=metadata+"Negative prompt: "+vs["ne"]+"\n"
	if "st" in vs:
		if vs["st"]!="":
			metadata=metadata+"Steps: "+vs["st"]+", " 
	if "sa" in vs:
		if vs["sa"]!="":
			metadata=metadata+"Sampler: "+vs["sa"]+", "
	if "cf" in vs:
		if vs["cf"]!="":
			metadata=metadata+"CFG scale: "+vs["cf"]+", "
	if "se" in vs:
		if vs["se"]!="":
			metadata=metadata+"Seed: "+vs["se"]+", "
	if "cl" in vs:
		if vs["cl"]!="":
			metadata=metadata+"Clip skip: "+vs["cl"]+", "
	if "ds" in vs:		
		if vs["ds"]!="":
			metadata=metadata+"Denoising strength: "+vs["ds"]+", "
	if "hu" in vs:
		if vs["hu"]!="":
			metadata=metadata+"Hires upscale: "+vs["hu"]+", "
	if "hs" in vs:
		if vs["hs"]!="":
			metadata=metadata+"Hires steps: "+vs["hs"]+", "
	if "hum" in vs:
		if vs["hum"]!="":
			metadata=metadata+"Hires upscaler: "+vs["hum"]+", "
	if "tu" in vs:
		if vs["tu"]!="":
			metadata=metadata+"Tile upscale: "+vs["tu"]+", "
	if "tum" in vs:
		if vs["tum"]!="":
			metadata=metadata+"Tile upscaler: "+vs["tum"]+", "
	if "ccs" in vs:
		if vs["ccs"]!="":
			metadata=metadata+"controlnet_conditioning_scale: "+vs["ccs"]+", "

	metadata=metadata+'Civitai resources: ['
	if "ckpt" in vs:
		if vs["ckpt"]!="":
			metadata=metadata+'{"type":"checkpoint","modelVersionId":'+vs["ckpt"]+"}"

	if "lora" in vs:
		if vs["lora"]!=[]:
			for i in range(len(vs["lora"])):
				metadata=metadata+',{"type":"lora","weight":'+str(vs["w"][i])+',"modelVersionId":'+str(vs["lora"][i])+"}"

	if "embed" in vs:
		if vs["embed"]!=[]:
			for i in range(len(vs["embed"])):
				metadata=metadata+',{"type":"embed","modelVersionId":'+str(vs["embed"][i])+"}"
	if "vae" in vs:
		if vs["vae"]!="":
			metadata=metadata+',{"type":"ae","modelVersionId":'+vs["vae"]+"}"
	if "cont" in vs:
		if vs["cont"]!="":
			metadata=metadata+',{"type":"controlnet","modelVersionId":'+vs["cont"]+"}"
	if "up" in vs:
		if vs["up"]!="":
			metadata=metadata+',{"type":"upscaler","modelVersionId":'+vs["up"]+"}"
			
	metadata=metadata+']'

	if "[," in metadata:
		metadata=metadata.replace("[,","[")
	try:
		image_path=vs["input"]
		if image_path.endswith(".png"):
			pnginfo = PngImagePlugin.PngInfo()
			pnginfo.add_text("parameters", metadata)
			img.save(image_path, "PNG", pnginfo=pnginfo)
		else:
			exif_data=piexif.helper.UserComment.dump(metadata, encoding="unicode")
			exif_dict={
				'Exif':{
					piexif.ExifIFD.UserComment:exif_data,
				}
			}
			exif_bytes = piexif.dump(exif_dict)
			img.save(image_path,"JPEG",quality = 85, exif=exif_bytes)
	except:
		image_path=vs["input"]
		if image_path.endswith(".png"):
			img.save(image_path, "PNG")
		else:
			img.save(image_path,"JPEG",quality = 85)

def get_id(path,w=None):
	if os.path.isdir(path):
		if os.path.exists(path+"/id.txt"):
			f=open(path+"/id.txt","r")
			meta_id=f.read()
			f.close()
		else:
			meta_id=""
		return meta_id
	try:
		f=safetensors.safe_open(path, framework="pt", device="cpu")
		meta_dict=f.metadata()
	except:
		meta_dict={}

	if "id" in meta_dict:
		meta_id=meta_dict["id"]
		if "," in meta_id:
			meta_id=meta_id.split(",")
			for i in range(len(meta_id)):
				try:
					meta_id[i]=int(meta_id[i])
				except:
					meta_id[i]=""
		else:
			try:
				meta_id=[int(meta_id)]
			except:
				meta_id=[""]
	else:
		meta_id=[""]

	if w==None:
		if type(meta_id)==list:
			meta_id=str(meta_id[0])
		return meta_id

	if "weight" in meta_dict:
		meta_weight=meta_dict["weight"]
		if "," in meta_weight:
			meta_weight=meta_weight.split(",")
			for i in range(len(meta_weight)):
				try:
					meta_weight[i]=float(meta_weight[i])*w
				except:
					meta_weight[i]=w
		else:
			try:
				meta_weight=[float(meta_weight)*w]
			except:
				meta_weight=[w]
	else:
		meta_weight=[w]

	while True:
		if len(meta_weight)>=len(meta_id):
			break
		meta_weight.appned(w)

	meta_id2=[]
	meta_weight2=[]
	for i in range(len(meta_id)):
		if meta_id[i]!="":
			meta_id2.append(meta_id[i])
			meta_weight2.append(meta_weight[i])

	return meta_id2,meta_weight2
	
def read_meta(path):
	out_sd={}
	if path.format=="JPEG":
		exif_data=path._getexif()
		exif_data=exif_data[37510].decode()
	else:
		exif_data = path.info['parameters']

	exif_data=exif_data.split("\n")
	exif_data2=str(exif_data.pop(-1).encode())
	k=None
	for i in range(len(exif_data)):
		if "\x00" in exif_data[i]:
			exif_data[i]=exif_data[i].replace("\x00","")
		if exif_data[i].startswith("Negative prompt: "):
			k=i
	if k==None:
		pro="".join(exif_data)
		neg=""
	else:
		pro="".join(exif_data[:k])
		neg="".join(exif_data[k:])
		neg=neg.replace("Negative prompt: ","")
	pro=pro.removeprefix("UNICODE")
	out_sd["pr"]=pro
	out_sd["ne"]=neg
	inds={
		"Steps:":"st",
		"CFG scale:":"cf",
		"Seed:":"se",
		"Clip skip:":"cl",
		"Denoising strength:":"ds",
		"Hires upscale:":"hu",
		"Hires steps:":"hs",
		"Hires upscaler:":"hum",
		"Tile upscale:":"tu",
		"Tile upscaler:":"tum",
		"controlnet_conditioning_scale:":"ccs"
	}
	ss=["karras","beta","exponential","sgm_uniform","simple","uniform","normal"]
	exif_data=exif_data2.replace(r"\x00","").removesuffix("'").removeprefix("b'").split(", ")
	for line in exif_data:
		for ind in inds:
			if line.startswith(ind):
				line2=line.split(": ")
				out_sd[inds[ind]]=line2[1]
		if line.startswith("Sampler:"):
			line2=line.split(": ")
			out_sd["sa1"]=line2[1]
			out_sd["sa2"]=""
			for ind in ss:
				if ind in line2[1].lower():
					out_sd["sa2"]=ind
					e=line2[1].split(" ")[-1]
					out_sd["sa1"]=line2[1].replace(" "+e,"")
		if line.startswith("Civitai resources:"):
			line2=line.replace("Civitai resources: ","")
			line2 = ast.literal_eval(line2)
			k1=1
			k2=1
			for sd in line2:
				if sd["type"]=="checkpoint":
					out_sd["ckpt"]=str(sd["modelVersionId"])
				elif sd["type"]=="lora":
					out_sd["lora"+str(k1)]=str(sd["modelVersionId"])
					out_sd["w"+str(k1)]=str(sd["weight"])
					if k1>4:
						del out_sd["lora"+str(k1)],out_sd["w"+str(k1)]
					k1=k1+1
				elif sd["type"]=="embed":
					out_sd["embed"+str(k2)]=str(sd["modelVersionId"])
					if k2>4:
						del out_sd["embed"+str(k2)]
					k2=k2+1
				elif sd["type"]=="controlnet":
					out_sd["cont"]=str(sd["modelVersionId"])
				elif sd["type"]=="upscaler":
					out_sd["up"]=str(sd["modelVersionId"])
				else:
					out_sd["vae"]=str(sd["modelVersionId"])
	return out_sd