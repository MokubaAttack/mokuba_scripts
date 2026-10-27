import torch
import FreeSimpleGUI as sg
import pyperclip
import tkinter as tk
import json
import os
import threading
from accelerate.commands.launch import (
	launch_command_parser,
	launch_command
)
import time

from ..tool.civitai_dl import check
from ..common.yaml import mk_yaml

class animagui:
	default_values={
		"pr":"",
		"ne":"",
		"n":"10",
		"st":"32",
		"sa":"flowmatch_euler","sc":"uniform",
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
	}
	default_json=os.getcwd()+"/pipecache/anima_default.json"

	def mokuani2(
		self,values,win,
	):
		win["RUN"].Update(disabled=True)
		start=time.time()
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

		if values["ds"]=="" or values["hu"]=="" or values["hum"]=="" or values["hs"]=="":
			mode=0
			ss=0.4
			step2=16
			up=1.5
			values["ds"]=""
			win["ds"].update("")
			values["hu"]=""
			win["hu"].update("")
			values["hs"]=""
			win["hs"].update("")
		else:
			mode=1
			try:
				ss=float(values["ds"])
			except:
				ss=0.4
			values["ds"]=str(ss)
			win["ds"].update(str(ss))
			try:
				up=float(values["hu"])
			except:
				up=1.5
			values["hu"]=str(up)
			win["hu"].update(str(up))
			try:
				step2=int(values["hs"])
			except:
				step2=16
			values["hs"]=str(step2)
			win["hs"].update(str(step2))

		try:
			with open(os.getcwd()+"/pipecache/token.json","r") as f:
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
		}
		f=open("pipecache/params.json","w")
		json.dump(params,f,indent=2)
		f.close()
		self.savevalues(values,values["out"]+"_params.json")

		gpu_num=int(values["gpu_number"])
		path=__file__.replace("\\","/")
		path1=path.replace("gui/animagui.py","run/anima_run.py")
		path2=mk_yaml(gpu_num)
		parser = launch_command_parser()
		args = parser.parse_args(["--config_file",path2,"--main_process_port","0",path1])
		try:
			launch_command(args)
		except Exception as e:
			print(e)

		if os.path.exists("pipecache/params.json"):
			os.remove("pipecache/params.json")
		if os.path.exists(path2):
			os.remove(path2)
		end=time.time()
		process_min=int(round(end-start)/60)
		process_sec=round(end-start)-60*process_min
		print(str(process_min)+"min"+str(process_sec)+"sec")
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
			"input","pr","ne","st","cf","se","n","x","y","lora1","lora2","lora3","lora4","w1","w2","w3","w4","hu","hs","hum","ds","sa","sc","out","gpu_number"
		]
	
		sa_list=["FlowMatch_Euler","FlowMatch_LCM"]
		sc_list=["karras","beta","exponential","normal"]
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
		]
		col4=[
			[sg.Text("width"), sg.Input(iv["x"],key="x",right_click_menu=grp_rclick_menu["x"], size=(10, 1))],
			[sg.Text("height"), sg.Input(iv["y"],key="y",right_click_menu=grp_rclick_menu["y"], size=(10, 1))],
			[sg.Text("Denoising strength"), sg.Input(iv["ds"],key="ds",right_click_menu=grp_rclick_menu["ds"], size=(10, 1))],
			[sg.Text("Hires upscale"), sg.Input(iv["hu"],key="hu",right_click_menu=grp_rclick_menu["hu"], size=(10, 1))],
			[sg.Text("Hires steps"), sg.Input(iv["hs"],key="hs",right_click_menu=grp_rclick_menu["hs"], size=(10, 1))],
			[sg.Text("Hires upscaler"), sg.Combo(default_value=iv["hum"],key="hum",values=hum_list,enable_events=True)],
		]
	
		layout=[
			[
				sg.Text("ckpt"),
				sg.Input(iv["input"],key="input",right_click_menu=grp_rclick_menu["input"]),
				sg.FileBrowse("File Browse",file_types=(('ckpt file', '.safetensors'),),key="file_browse", enable_events=True),
				sg.FolderBrowse("Folder Browse",key="folder_browse", enable_events=True)
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
			[sg.Column(col1),sg.Column(col2)],
			[sg.Column(col3),sg.Column(col4)],
			[
				sg.Text("dtype"), sg.Combo(default_value=iv["dtype"],values=["f32","f16","bf16"],key="dtype"),
				sg.Text("gpu number"), sg.Input("1",key="gpu_number",right_click_menu=grp_rclick_menu["gpu_number"], size=(10, 1))
			],
			[
				sg.Text("output folder"),
				sg.Input(iv["out"],key="out",right_click_menu=grp_rclick_menu["out"]),sg.FolderBrowse()
			],
			[sg.Button("Save Params",key="save"),sg.Button("Load Params",key="load",disabled=d),sg.Button("Save Params to",key="save2"),sg.Button("Load Params from",key="load2")],
			[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
		]
	
		window = sg.Window('anima gui', layout)
	
		while True:
			event, values = window.read()
			if event == sg.WINDOW_CLOSED:
				break
			elif event=="EXIT":
				break
			elif event=="RUN":
				if values["input"]!="" and values["out"]!="":
					thread1 = threading.Thread(target=self.mokuani2,args=(values,window))
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
			elif event=="folder_browse":
				window["input"].update(values["folder_browse"])
			elif event=="file_browse":
				window["input"].update(values["file_browse"])
	
		window.close()
		
