import safetensors
from PIL import PngImagePlugin
import piexif
import piexif.helper
import os

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
		if vs["lora"]!="[]":
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