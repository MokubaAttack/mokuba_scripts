import torch
import FreeSimpleGUI as sg
import pyperclip
import tkinter as tk
import json
import os
import time
import re

from ..sd.mokusdxlpipe import mokusdxlpipe
from ..common.discord import up_drop
from ..common.flush import flush
from ..common.metadata import plus_meta
from ..common.seed import make_seed
from ..common.dl import dlc
from ..tool.civitai_dl import check

class sdxlgui:
	default_values={
		"pr":"",
		"ne":"",
		"n":"10",
		"st":"30",
		"sa":"DDIM","sc":"",
		"cf":"4",
		"se":"0",
		"x":"896",
		"y":"1152",
		"ds":"0.4",
		"hu":"1.5",
		"hs":"16",
		"hum":"NEAREST",
		"lora1":"","w1":"1.0",
		"lora2":"","w2":"1.0",
		"lora3":"","w3":"1.0",
		"lora4":"","w4":"1.0",
		"input":"",
		"out":"",
		"dtype":"bf16","dev":"xpu",
		"vae":"",
		"p1":"","n1":"",
		"p2":"","n2":"",
		"p3":"","n3":"",
		"cl":"2",
		"tu":"1.33",
		"ts":"20",
		"ccs":"1.0",
		"lowmem":True,
	}
	default_json=os.getcwd()+"/sdxl_default.json"

	def mokusdxl(
		self,
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

		try:
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
		except Exception as e:
			return e

		if not(os.path.exists(out_folder+"/0")):
			os.makedirs(out_folder+"/0")

		imgs=[]
		for i,s in enumerate(seed):
			try:
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
			except Exception as e:
				return e

		if mode==0:
			if url!=[]:
				try:
					shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
					drop_path=str(ut)+".zip"
					up_drop("archive_shutil.zip",drop_path,url)
					os.remove("archive_shutil.zip")
				except Exception as e:
					return e
			return "fin"

		if not(os.path.exists(out_folder+"/1")):
			os.makedirs(out_folder+"/1")

		try:
			pipe.make_upscaler(Interpolation,dev)
			x=round(x*up/8)*8
			y=round(y*up/8)*8
			for i,img in enumerate(imgs):
				imgs[i]=pipe.upscale(img,x,y)
			pipe.delete_upscaler()
		except Exception as e:
			return e

		for i,s in enumerate(seed):
			try:
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
			except Exception as e:
				return e

		if mode==1:
			if url!=[]:
				try:
					shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
					drop_path=str(ut)+".zip"
					up_drop("archive_shutil.zip",drop_path,url)
					os.remove("archive_shutil.zip")
				except Exception as e:
					return e
			return pipe

		if not(os.path.exists(out_folder+"/2")):
			os.makedirs(out_folder+"/2")

		try:
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
		except Exception as e:
			return e

		for i,s in enumerate(seed):
			try:
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
			except Exception as e:
				return e

		if url!=[]:
			try:
				shutil.make_archive('archive_shutil', format='zip', root_dir=out_folder)
				drop_path=str(ut)+".zip"
				up_drop("archive_shutil.zip",drop_path,url)
				os.remove("archive_shutil.zip")
			except Exception as e:
				return e
		return "fin"

	def setvalues(self):
		iv=self.default_values
		e=True
		if os.path.exists(self.default_json):
			f=open(self.default_json,"r")
			sd=json.load(f)
			f.close()
			for k in iv:
				if k in sd:
					iv[k]=sd[k]
			e=False
		return iv,e

	def loadvalues(self,win,path=None):
		if path==None:
			f=open(self.default_json,"r")
		else:
			f=open(path,"r")
		sd=json.load(f)
		f.close()
		iv=self.default_values
		for k in iv:
			if k in sd:
				win[k].update(sd[k])
			else:
				win[k].update(iv[k])
	
	def savevalues(self,vs,path=None):
		iv=self.default_values
		for k in iv:
			iv[k]=vs[k]
		if path==None:
			f=open(self.default_json,"w")
		else:
			f=open(path,"w")
		json.dump(iv,f)
		f.close()

	def gui(self):
		if check()==False:
			return
		iv,d=self.setvalues()

		keys=[
			"input","pr","ne","st","cf","se","n","x","y","lora1","lora2","lora3","lora4","w1","w2","w3","w4","hu","hs","hum","ds","sa","sc","out",
			"vae","p1","p2","p3","n1","n2","n3","cl","tu","ts","ccs"
		]

		sa_list=[
			"Euler a",
			"Euler",
			"LMS",
			"Heun",
			"DPM2",
			"DPM2 a",
			"DPM++",
			"DPM++ 2M",
			"DPM++ SDE",
			"DPM++ 2M SDE",
			"DPM++ 3M SDE",
			"DDIM",
			"PLMS",
			"UniPC",
			"LCM"
		]
		sc_list=["","Karras","beta","exponential","sgm_uniform","simple"]
		hum_list=["NEAREST","BOX","BILINEAR","HAMMING","BICUBIC","LANCZOS","select file"]

		grp_rclick_menu={}
		for key in keys:
			grp_rclick_menu[key]=[
				"",
				[
					"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
				]
			] 

		col1=[
			[sg.Text("prompt")],
			[sg.Multiline(iv["pr"],size=(50, 5),key="pr",right_click_menu=grp_rclick_menu["pr"])]
		]
		col2=[
			[sg.Text("negative prompt")],
			[sg.Multiline(iv["ne"],size=(50, 5),key="ne",right_click_menu=grp_rclick_menu["ne"])]
		]
		col3=[
			[sg.Text("Pic number"), sg.Input(iv["n"],key="n",right_click_menu=grp_rclick_menu["n"], size=(10, 1))],
			[sg.Text("Steps"), sg.Input(iv["st"],key="st",right_click_menu=grp_rclick_menu["st"], size=(10, 1))],
			[sg.Text("Sampler"), sg.Combo(default_value=iv["sa"],values=sa_list,key="sa")],
			[sg.Text("Schedule type"), sg.Combo(default_value=iv["sc"],values=sc_list,key="sc")],
			[sg.Text("CFG scale"), sg.Input(iv["cf"],key="cf",right_click_menu=grp_rclick_menu["cf"], size=(10, 1))],
			[sg.Text("Seed"), sg.Input(iv["se"],key="se",right_click_menu=grp_rclick_menu["se"], size=(20, 1))],
			[sg.Text("Clip skip"), sg.Input(iv["cl"],key="cl",right_click_menu=grp_rclick_menu["cl"], size=(10, 1))],
		]
		col4=[
			[sg.Text("width"), sg.Input(iv["x"],key="x",right_click_menu=grp_rclick_menu["x"], size=(10, 1))],
			[sg.Text("height"), sg.Input(iv["y"],key="y",right_click_menu=grp_rclick_menu["y"], size=(10, 1))],
			[sg.Text("Denoising strength"), sg.Input(iv["ds"],key="ds",right_click_menu=grp_rclick_menu["ds"], size=(10, 1))],
			[sg.Text("Hires upscale"), sg.Input(iv["hu"],key="hu",right_click_menu=grp_rclick_menu["hu"], size=(10, 1))],
			[sg.Text("Hires steps"), sg.Input(iv["hs"],key="hs",right_click_menu=grp_rclick_menu["hs"], size=(10, 1))],
			[sg.Text("Tile upscale"), sg.Input(iv["tu"],key="tu",right_click_menu=grp_rclick_menu["tu"], size=(10, 1))],
			[sg.Text("Tile steps"), sg.Input(iv["ts"],key="ts",right_click_menu=grp_rclick_menu["ts"], size=(10, 1))],
			[sg.Text("Upscaler"), sg.Combo(default_value=iv["hum"],key="hum",values=hum_list,enable_events=True)],
			[sg.Text("controlnet_conditioning_scale"), sg.Input(iv["ccs"],key="ccs",right_click_menu=grp_rclick_menu["ccs"], size=(10, 1))],
		]
		col5=[
			[sg.Text("positive embedding")],
			[sg.Input(iv["p1"],key="p1",right_click_menu=grp_rclick_menu["p1"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
			[sg.Input(iv["p2"],key="p2",right_click_menu=grp_rclick_menu["p2"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
			[sg.Input(iv["p3"],key="p3",right_click_menu=grp_rclick_menu["p3"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
		]
		col6=[
			[sg.Text("negative embedding")],
			[sg.Input(iv["n1"],key="n1",right_click_menu=grp_rclick_menu["n1"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
			[sg.Input(iv["n2"],key="n2",right_click_menu=grp_rclick_menu["n2"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
			[sg.Input(iv["n3"],key="n3",right_click_menu=grp_rclick_menu["n3"]),sg.FilesBrowse(file_types=(('embedding file', '.safetensors'),))],
		]
		col7=[
			[sg.Button("Save Params",key="save")],
			[sg.Button("Load Params",key="load",disabled=d)],
			[sg.Button("Save Params to",key="save2")],
			[sg.Button("Load Params from",key="load2")],
			[sg.Button('RUN', key='RUN')],
			[sg.Button('EXIT', key='EXIT')]
		]

		layout=[
			[
				sg.Text("ckpt"),
				sg.Input(iv["input"],key="input",right_click_menu=grp_rclick_menu["input"]),sg.FileBrowse(file_types=(('ckpt file', '.safetensors'),)),
			],
			[
				sg.Text("lora1"),
				sg.Input(iv["lora1"],key="lora1",right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse(file_types=(('lora file', '.safetensors'),)),
				sg.Text("weight"),
				sg.Input(iv["w1"],key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))
			],
			[
				sg.Text("lora2"),
				sg.Input(iv["lora2"],key="lora2",right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse(file_types=(('lora file', '.safetensors'),)),
				sg.Text("weight"),
				sg.Input(iv["w2"],key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))
			],
			[
				sg.Text("lora3"),
				sg.Input(iv["lora3"],key="lora3",right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse(file_types=(('lora file', '.safetensors'),)),
				sg.Text("weight"),
				sg.Input(iv["w3"],key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))
			],
			[
				sg.Text("lora4"),
				sg.Input(iv["lora4"],key="lora4",right_click_menu=grp_rclick_menu["lora4"]),sg.FileBrowse(file_types=(('lora file', '.safetensors'),)),
				sg.Text("weight"),
				sg.Input(iv["w4"],key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))
			],
			[
				sg.Text("vae"),
				sg.Input(iv["vae"],key="vae",right_click_menu=grp_rclick_menu["vae"]),sg.FileBrowse(file_types=(('vae file', '.safetensors'),)),
			],
			[sg.Column(col5),sg.Column(col6)],
			[sg.Column(col1),sg.Column(col2)],
			[sg.Column(col3),sg.Column(col4),sg.Column(col7)],
			[
				sg.Text("dtype"), sg.Combo(default_value=iv["dtype"],values=["f32","f16","bf16"],key="dtype"),
				sg.Text("device"), sg.Combo(default_value=iv["dev"],values=["cpu","cuda","mps","xpu"],key="dev"),
				sg.Checkbox('gpu low memory', key='lowmem',default=iv["lowmem"])
			],
			[
				sg.Text("output folder"),
				sg.Input(iv["out"],key="out",right_click_menu=grp_rclick_menu["out"]),sg.FolderBrowse()
			]
		]

		window = sg.Window('sdxl gui', layout)

		while True:
			event, values = window.read()
			if event == sg.WINDOW_CLOSED:
				values={}
				break
			elif event=="EXIT":
				values={}
				break
			elif event=="RUN":
				if values["input"]!="" and values["out"]!="":
					break
			elif "-copy-" in event:
				try:
					key=event.replace("-copy-::","")
					selected = window[key].widget.selection_get()
					pyperclip.copy(selected)
				except:
					pass
			elif "-cut-" in event:
				try:
					key=event.replace("-cut-::","")
					selected = window[key].widget.selection_get()
					pyperclip.copy(selected)
					window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
				except:
					pass
			elif "-paste-" in event:
				try:
					key=event.replace("-paste-::","")
					selected = pyperclip.paste()
					insert_pos = window[key].widget.index("insert")
					window[key].Widget.insert(insert_pos, selected)
					window[key].widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
				except:
					pass
			elif "hum" in event:
				if values["hum"]=="select file":
					value = sg.popup_get_file('upscaler file',file_types=(('upscaler File', '.pth'),))
					if value!=None:
						window["hum"].update(value)
			elif event=="save":
				try:
					self.savevalues(values)
					window["load"].update(disabled=False)
				except:
					pass
			elif event=="load":
				try:
					self.loadvalues(window)
				except:
					pass
			elif event=="save2":
				try:
					value = sg.popup_get_file('save file',file_types=(('json File', '.json'),),save_as=True)
					if value!=None:
						self.savevalues(values,value)
				except:
					pass
			elif event=="load2":
				try:
					value = sg.popup_get_file('load file',file_types=(('json File', '.json'),))
					if value!=None:
						self.loadvalues(window,value)
				except:
					pass

		window.close()

		if values!={}:
			self.savevalues(values,values["out"]+"_params.json")
		
			start_time=time.time()

			loras=[]
			lora_weights=[]
			for i in range(4):
				if values["lora"+str(i+1)]!="":
					loras.append(values["lora"+str(i+1)])
					try:
						lora_weights.append(float(values["w"+str(i+1)]))
					except:
						lora_weights.append(1.0)

			pos_emb=[]
			neg_emb=[]
			for i in range(3):
				if values["p"+str(i+1)]!="":
					pos_emb.append(values["p"+str(i+1)])
				if values["n"+str(i+1)]!="":
					neg_emb.append(values["n"+str(i+1)])
			
			try:
				n=int(values["n"])
			except:
				n=10

			try:
				gs=float(values["cf"])
			except:
				gs=7

			try:
				step=int(values["st"])
			except:
				step=30
			try:
				step2=int(values["hs"])
			except:
				step2=15
			try:
				step3=int(values["ts"])
			except:
				step3=20

			try:
				x=int(values["x"])
			except:
				x=1024
			try:
				y=int(values["y"])
			except:
				y=1024

			try:
				ccs=float(values["ccs"])
			except:
				ccs=1.0

			try:
				cs=int(values["cl"])
			except:
				cs=2

			try:
				ss=float(values["ds"])
			except:
				ss=0.4
				
			try:
				up=float(values["hu"])
			except:
				up=1.5
			try:
				up2=float(values["tu"])
			except:
				up2=1.33
				
			mode=0
			if values["ds"]!="" and values["hs"]!="" and values["hu"]!="":
				mode=1
				if values["tu"]!="" and values["ts"]!="" and values["ccs"]!="":
					mode=2

			try:
				with open(os.getcwd()+"/token.json","r") as f:
					d = json.load(f)
				token=d["civitai_token"]
			except:
				token=""
			
			result=self.mokusdxl(
				loras=loras,
				lora_weights=lora_weights,
				prompt = values["pr"],
				n_prompt = values["ne"],
				pic_number=n,
				gs=gs,
				sample=values["sa"],
				sgm=values["sc"],
				seed=values["se"],
				out_folder=values["out"],
				base_safe=values["input"],
				url=[],
				dtype=values["dtype"],
				dev=values["dev"],
				Interpolation=values["hum"],
				step=step,
				step2=step2,
				step3=step3,
				ss=ss,
				cs=cs,
				gpulowmem=values["lowmem"],
				freezeunet=values["lowmem"],
				vae_safe=values["vae"],
				pos_emb=pos_emb,
				neg_emb=neg_emb,
				ccs=ccs,
				up=up,
				up2=up2,
				x=x,
				y=y,
				mode=mode,
				token=token,
				)
			end_time=time.time()
			time_sec=round(end_time-start_time)
			time_min=int(time_sec/60)
			time_sec=time_sec-60*time_min
			result=str(result)+"\n"+str(time_min)+"min"+str(time_sec)+"sec"
			sg.popup(result,title='sdxl gui')

