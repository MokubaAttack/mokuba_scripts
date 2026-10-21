import torch
import FreeSimpleGUI as sg
import pyperclip
import tkinter as tk
import json
import os
import threading

from ..tool.civitai_dl import check
from ..common.yaml import mk_yaml

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
		"dtype":"bf16",
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

	def mokusdxl2(
		self,values,win,
	):
		win["RUN"].Update(disabled=True)
		loras=[]
		lora_weights=[]
		for i in range(4):
			if values["lora"+str(i+1)]!="":
				loras.append(values["lora"+str(i+1)])
				try:
					lora_weights.append(float(values["w"+str(i+1)]))
				except:
					lora_weights.append(1.0)
				win["lora"+str(i+1)].update(str(loras[-1]))
				values["lora"+str(i+1)]=str(loras[-1])
				win["w"+str(i+1)].update(str(lora_weights[-1]))
				values["w"+str(i+1)]=str(lora_weights[-1])

		pos_emb=[]
		neg_emb=[]
		for i in range(3):
			if values["p"+str(i+1)]!="":
				pos_emb.append(values["p"+str(i+1)])
				win["p"+str(i+1)].update(str(pos_emb[-1]))
				values["p"+str(i+1)]=str(pos_emb[-1])
			if values["n"+str(i+1)]!="":
				neg_emb.append(values["n"+str(i+1)])
				win["n"+str(i+1)].update(str(neg_emb[-1]))
				values["n"+str(i+1)]=str(neg_emb[-1])
	
		try:
			n=int(values["n"])
		except:
			n=10
		win["n"].update(str(n))
		values["n"]=str(n)

		try:
			gs=float(values["cf"])
		except:
			gs=4
		win["cf"].update(str(gs))
		values["cf"]=str(gs)

		try:
			step=int(values["st"])
		except:
			step=32
		values["st"]=str(step)
		win["st"].update(str(step))

		try:
			cs=int(values["cl"])
		except:
			cs=2
		values["cl"]=str(cs)
		win["cl"].update(str(cs))

		try:
			x=int(values["x"])
		except:
			x=1024
		try:
			y=int(values["y"])
		except:
			y=1024
		values["x"]=str(x)
		win["x"].update(str(x))
		values["y"]=str(y)
		win["y"].update(str(y))

		if values["ds"]!="" and values["hs"]!="" and values["hu"]!="" and values["tu"]!="" and values["ts"]!="" and values["ccs"]!="":
			mode=2
			try:
				step2=int(values["hs"])
			except:
				step2=15
			values["hs"]=str(step2)
			win["hs"].update(str(step2))
			try:
				up=float(values["hu"])
			except:
				up=1.5
			values["hu"]=str(up)
			win["hu"].update(str(up))
			try:
				ss=float(values["ds"])
			except:
				ss=0.4
			values["ds"]=str(ss)
			win["ds"].update(str(ss))
			try:
				up2=float(values["tu"])
			except:
				up2=1.33
			values["tu"]=str(up2)
			win["tu"].update(str(up2))
			try:
				ccs=float(values["ccs"])
			except:
				ccs=1.0
			values["ccs"]=str(ccs)
			win["ccs"].update(str(ccs))
			try:
				step3=int(values["ts"])
			except:
				step3=20
			values["ts"]=str(step3)
			win["ts"].update(str(step3))
		elif values["ds"]!="" and values["hs"]!="" and values["hu"]!="":
			mode=1
			ccs=1.0
			up2=1.33
			step3=20
			try:
				step2=int(values["hs"])
			except:
				step2=15
			values["hs"]=str(step2)
			win["hs"].update(str(step2))
			try:
				up=float(values["hu"])
			except:
				up=1.5
			values["hu"]=str(up)
			win["hu"].update(str(up))
			try:
				ss=float(values["ds"])
			except:
				ss=0.4
			values["ds"]=str(ss)
			win["ds"].update(str(ss))
			values["tu"]=""
			win["tu"].update("")
			values["ts"]=""
			win["ts"].update("")
			values["ccs"]=""
			win["ccs"].update("")
		else:
			mode=0
			ss=0.4
			step2=16
			up=1.5
			ccs=1.0
			up2=1.33
			step3=20
			values["ds"]=""
			win["ds"].update("")
			values["hu"]=""
			win["hu"].update("")
			values["hs"]=""
			win["hs"].update("")
			values["tu"]=""
			win["tu"].update("")
			values["ts"]=""
			win["ts"].update("")
			values["ccs"]=""
			win["ccs"].update("")

		try:
			with open(os.getcwd()+"/token.json","r") as f:
				d = json.load(f)
			token=d["civitai_token"]
		except:
			token=""
		
		params={
		"loras":loras,
		"lora_weights":lora_weights,
		"prompt":values["pr"],
		"n_prompt":values["ne"],
		"pic_number":n,
		"gs":gs,
		"step":step,
		"sample":values["sa"],
		"sgm":values["sc"],
		"seed":values["se"],
		"out_folder":values["out"],
		"base_safe":values["input"],
		"url":[],
		"dtype":values["dtype"],
		"x":x,
		"y":y,
		"mode":mode,
		"up":up,
		"Interpolation":values["hum"],
		"step2":step2,
		"ss":ss,
		"token":token,
		"pos_emb":pos_emb,
		"neg_emb":neg_emb,
		"vae_safe":values["vae"],
		"step3":step3,
		"up2":up2,
		"ccs":ccs,
		"gpulowmem":values["lowmem"],
		"freezeunet":values["lowmem"],
		"cs":cs,
		"qprompt":"masterpiece, best quality, ultra detailed",
		"qn_prompt":"worst quality, low quality, normal quality",
		}
		f=open("params.json","w")
		json.dump(params,f,indent=2)
		f.close()
		self.savevalues(values,values["out"]+"_params.json")

		gpu_num=int(values["gpu_number"])
		path=__file__.replace("\\","/")
		path=path.replace("gui/sdxlgui.py","run/sdxl_run.py")
		path2=mk_yaml(gpu_num)
		output=subprocess.run("accelerate launch --config_file "+path2+" "+path, shell=True,check=False)

		if os.path.exists("params.json"):
			os.remove("params.json")
		if os.path.exists(path2):
			os.remove(path2)
		win["RUN"].Update(disabled=False)

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
			"input","pr","ne","st","cf","se","n","x","y","lora1","lora2","lora3","lora4","w1","w2","w3","w4","hu","hs","hum","ds","sa","sc","out","vae","p1","p2","p3","n1","n2","n3","cl","tu","ts","ccs","gpu_number"
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
				sg.Text("gpu number"), sg.Input("1",key="gpu_number",right_click_menu=grp_rclick_menu["gpu_number"], size=(10, 1)),
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
				break
			elif event=="EXIT":
				break
			elif event=="RUN":
				if values["input"]!="" and values["out"]!="":
					thread1 = threading.Thread(target=self.mokusdxl2,args=(values,window))
					thread1.start()
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
