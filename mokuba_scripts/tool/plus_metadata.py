import os
import json
from PIL import Image

from ..common.metadata import (
	plus_meta,
	read_meta
)

default_sd={
	"pr":"",
	"ne":"",
	"st":"30",
	"sa1":"DDIM",
	"sa2":"",
	"cf":"7",
	"se":"12345",
	"cl":"2",
	"ds":"0.4",
	"hu":"1.5",
	"hum":"NEAREST",
	"hs":"20",
	"tu":"1.5",
	"tum":"NEAREST",
	"ccs":"1.0",
	"ckpt":"",
	"lora":[],
	"w":[],
	"embed":[],
	"vae":"",
	"up":"",
	"cont":""
}
default_json=os.getcwd()+"/metadata_default.json"

def setvalues():
	if os.path.exists(default_json):
		f=open(default_json,"r")
		sd=json.load(f)
		f.close()
		e=False
	else:
		sd=default_sd
		e=True
	for i in range(4):
		sd["lora"+str(i+1)]=""
		sd["w"+str(i+1)]=""
		sd["embed"+str(i+1)]=""
	for i in range(len(sd["lora"])):
		sd["lora"+str(i+1)]=sd["lora"][i]
	for i in range(len(sd["w"])):
		sd["w"+str(i+1)]=sd["w"][i]
	for i in range(len(sd["embed"])):
		sd["embed"+str(i+1)]=sd["embed"][i]
	return sd,e
	
def loadvalues(win,path=default_json):
	if os.path.exists(path):
		f=open(default_json,"r")
		sd=json.load(f)
		f.close()
		for i in range(4):
			sd["lora"+str(i+1)]=""
			sd["w"+str(i+1)]=""
			sd["embed"+str(i+1)]=""
		for i in range(len(sd["lora"])):
			sd["lora"+str(i+1)]=sd["lora"][i]
		for i in range(len(sd["w"])):
			sd["w"+str(i+1)]=sd["w"][i]
		for i in range(len(sd["embed"])):
			sd["embed"+str(i+1)]=sd["embed"][i]
		del sd["lora"],sd["w"],sd["embed"]
		for key in sd:
			win[key].update(sd[key])

def savevalues(vs,path=default_json):
	sd=default_sd
	for key in sd:
		if key=="lora":
			for i in range(4):
				sd["lora"].append(vs["lora"+str(i+1)])
		elif key=="w":
			for i in range(4):
				sd["w"].append(vs["w"+str(i+1)])
		elif key=="embed":
			for i in range(4):
				sd["embed"].append(vs["embed"+str(i+1)])
		else:
			sd[key]=vs[key]
	f=open(path,"w")
	json.dump(sd,f)
	f.close()

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import pyperclip
	
	sg.theme('TealMono')
	
	ivs,e=setvalues()
	
	keys=[
		"input","pr","ne","st","sa1","cf","se","cl","ckpt","lora1","lora2","lora3","lora4","embed1","embed2","embed3","embed4","w1","w2","w3","w4","vae","hu","hs","hum","ds","sa2","tu","tum","up","cont","ccs"
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
		"LCM",
		"FlowMatch_Euler",
		"FlowMatch_LCM"
	]
	sc_list=[
		"","Karras","beta","exponential","sgm_uniform","simple","uniform","normal"
	]
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
		[sg.Multiline(ivs["pr"], size=(50, 5),key="pr",right_click_menu=grp_rclick_menu["pr"])]
	]
	col2=[
		[sg.Text("negative prompt")],
		[sg.Multiline(ivs["ne"], size=(50, 5),key="ne",right_click_menu=grp_rclick_menu["ne"])]
	]
	col3=[
		[sg.Text("Steps"), sg.Input(ivs["st"],key="st",right_click_menu=grp_rclick_menu["st"], size=(10, 1))],
		[sg.Text("Sampler"), sg.Combo(default_value=ivs["sa1"],values=sa_list,key="sa1")],
		[sg.Text("Schedule type"), sg.Combo(default_value=ivs["sa2"],values=sc_list,key="sa2")],
		[sg.Text("CFG scale"), sg.Input(ivs["cf"],key="cf",right_click_menu=grp_rclick_menu["cf"], size=(10, 1))],
		[sg.Text("Seed"), sg.Input(ivs["se"],key="se",right_click_menu=grp_rclick_menu["se"], size=(20, 1))],
		[sg.Text("Clip skip"), sg.Input(ivs["cl"],key="cl",right_click_menu=grp_rclick_menu["cl"], size=(10, 1))],
	]
	col4=[
		[sg.Text("Denoising strength"), sg.Input(ivs["ds"],key="ds",right_click_menu=grp_rclick_menu["ds"], size=(10, 1))],
		[sg.Text("Hires upscale"), sg.Input(ivs["hu"],key="hu",right_click_menu=grp_rclick_menu["hu"], size=(10, 1))],
		[sg.Text("Hires steps"), sg.Input(ivs["hs"],key="hs",right_click_menu=grp_rclick_menu["hs"], size=(10, 1))],
		[sg.Text("Hires upscaler"), sg.Combo(default_value=ivs["hum"],key="hum",values=hum_list,enable_events=True)],
		[sg.Text("Tile upscale"), sg.Input(ivs["tu"],key="tu",right_click_menu=grp_rclick_menu["tu"], size=(10, 1))],
		[sg.Text("Tile upscaler"), sg.Combo(default_value=ivs["tum"],key="tum",values=hum_list,enable_events=True)],
		[sg.Text("controlnet_conditioning_scale"), sg.Input(ivs["ccs"],key="ccs",right_click_menu=grp_rclick_menu["ccs"], size=(10, 1))],
	]
		
	layout=[
		[
			sg.Text("input"),
			sg.Input(key="input",right_click_menu=grp_rclick_menu["input"]),sg.FileBrowse(file_types=(('image file', '.png'),('image file', '.jpg'))),
			sg.Button('READ', key='READ'),
		],
		[sg.Column(col1),sg.Column(col2)],
		[sg.Column(col3),sg.Column(col4)],
		[sg.Text("ckpt modelVersionId"), sg.Input(ivs["ckpt"],key="ckpt",right_click_menu=grp_rclick_menu["ckpt"], size=(20, 1)),sg.Text("vae modelVersionId"), sg.Input(ivs["vae"],key="vae",right_click_menu=grp_rclick_menu["vae"], size=(20, 1))],
		[sg.Text("lora1 modelVersionId"), sg.Input(ivs["lora1"],key="lora1",right_click_menu=grp_rclick_menu["lora1"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w1"],key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))],
		[sg.Text("lora2 modelVersionId"), sg.Input(ivs["lora2"],key="lora2",right_click_menu=grp_rclick_menu["lora2"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w2"],key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))],
		[sg.Text("lora3 modelVersionId"), sg.Input(ivs["lora3"],key="lora3",right_click_menu=grp_rclick_menu["lora3"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w3"],key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))],
		[sg.Text("lora4 modelVersionId"), sg.Input(ivs["lora4"],key="lora4",right_click_menu=grp_rclick_menu["lora4"], size=(20, 1)),sg.Text("weight"), sg.Input(ivs["w4"],key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))],
		[sg.Text("embed1 modelVersionId"), sg.Input(ivs["embed1"],key="embed1",right_click_menu=grp_rclick_menu["embed1"], size=(20, 1)),sg.Text("embed2 modelVersionId"), sg.Input(ivs["embed2"],key="embed2",right_click_menu=grp_rclick_menu["embed2"], size=(20, 1))],
		[sg.Text("embed3 modelVersionId"), sg.Input(ivs["embed3"],key="embed3",right_click_menu=grp_rclick_menu["embed3"], size=(20, 1)),sg.Text("embed4 modelVersionId"), sg.Input(ivs["embed4"],key="embed4",right_click_menu=grp_rclick_menu["embed4"], size=(20, 1))],
		[sg.Text("controlnet modelVersionId"), sg.Input(ivs["cont"],key="cont",right_click_menu=grp_rclick_menu["cont"], size=(20, 1)),sg.Text("upscaler modelVersionId"), sg.Input(ivs["up"],key="up",right_click_menu=grp_rclick_menu["up"], size=(20, 1))],
		[sg.Text("infomation",key="info")],
		[sg.Button("Save Params",key="save"),sg.Button("Load Params",key="load",disabled=e),sg.Button("Save Params to",key="save2"),sg.Button("Load Params from",key="load2")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Plus Metadata', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["input"]!="" and values["ckpt"]!="":
				try:
					image = Image.open(values["input"])
					sd=values
					if sd["input"].endswith(".jpg"):
						sd["input"]=sd["input"].removesuffix(".jpg")+"_meta.jpg"
					else:
						sd["input"]=sd["input"].removesuffix(".png")+"_meta.png"
					sd["lora"]=[]
					sd["w"]=[]
					sd["embed"]=[]
					for i in range(4):
						if sd["lora"+str(i+1)]!="":
							sd["lora"].append(sd["lora"+str(i+1)])
							sd["w"].append(sd["w"+str(i+1)])
						if sd["embed"+str(i+1)]!="":
							sd["embed"].append(sd["embed"+str(i+1)])
						del sd["lora"+str(i+1)],sd["w"+str(i+1)],sd["embed"+str(i+1)]
					plus_meta(sd,image)
					window["info"].update("fin : "+os.path.basename(sd["input"]))
				except:
					window["info"].update("error")
					
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
		elif event=="save":
			try:
				savevalues(values)
				window["load"].update(disabled=False)
			except:
				pass
		elif event=="load":
			try:
				loadvalues(window)
			except:
				pass
		elif event=="save2":
			try:
				value = sg.popup_get_file('save file',file_types=(('json File', '.json'),),save_as=True)
				if value!=None:
					savevalues(values,value)
			except:
				pass
		elif event=="load2":
			try:
				value = sg.popup_get_file('load file',file_types=(('json File', '.json'),))
				if value!=None:
					loadvalues(window,value)
			except:
				pass
		elif event=="READ":
			if values["input"]!="":
				try:
					image = Image.open(values["input"])
					sd=read_meta(image)
					for key in ["pr","ne","st","sa1","cf","se","cl","ckpt","lora1","lora2","lora3","lora4","embed1","embed2","embed3","embed4","w1","w2","w3","w4","vae","hu","hs","hum","ds","sa2","tu","tum","up","cont","ccs"]:
						if key in sd:
							window[key].update(sd[key])
						else:
							window[key].update("")
					window["info"].update("read : "+os.path.basename(values["input"]))
				except:
					window["info"].update("error")
		elif "hum" in event:
			if values["hum"]=="select file":
				value = sg.popup_get_file('upscaler file',file_types=(('upscaler File', '.pth'),))
				if value!=None:
					window["hum"].update(os.path.basename(value))
				else:
					window["hum"].update("NEAREST")
		elif "tum" in event:
			if values["tum"]=="select file":
				value = sg.popup_get_file('upscaler file',file_types=(('upscaler File', '.pth'),))
				if value!=None:
					window["tum"].update(os.path.basename(value))
				else:
					window["tum"].update("NEAREST")
	window.close()
