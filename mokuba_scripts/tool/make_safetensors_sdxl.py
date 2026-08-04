import os
import shutil
import torch
from safetensors.torch import (
	save_file,
	load_file
)
import math

from ..sd.mokusdxlpipe import mokusdxlpipe
from ..common.diff2 import diff2sdxl
from ..common.save_safe import save_safe
from ..common.flush import flush

def zip_ckpt(sd1,sd2):
	keys=list(sd1)
	for k in keys :
		t=sd1.pop(k).to(torch.float32)
		if k in sd2:
			sum1=torch.sum(torch.abs(t)).item()
			sum2=torch.sum(torch.abs(sd2.pop(k).to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t=t*sum2/sum1

		osd={}
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			
	return keys

def makesafe(base_path,loras,ws,out_path,vae,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)
		
	if win!=None:
		win["info"].update("make pipe")
	else:
		print("make pipe")
	
	try:
		pipe=mokusdxlpipe()
		if base_path.endswith(".safetensors"):
			pipe.from_safe(path=base_path,vae_path=vae)
		else:
			pipe.from_folder(path=base_path,vae_path=vae)
	except:
		if win!=None:
			win["RUN"].Update(disabled=False)
			win["info"].update("Unsupported sdxl checkpoint")
		else:
			print("Unsupported sdxl checkpoint")
		return
	flush()

	if win!=None:
		win["info"].update("merge lora")
	else:
		print("merge lora")

	for i in range(len(loras)):
		try:
			pipe.load_lycoris(path=loras[i],weight=ws[i])
		except:
			if win!=None:
				win["RUN"].Update(disabled=False)
				win["info"].update(loras[i]+" isn't supported.")
			else:
				print(loras[i]+" isn't supported.")
			return
	flush()

	if os.path.exists(os.getcwd()+"/safe_temp"):
		shutil.rmtree(os.getcwd()+"/safe_temp")
	os.mkdir(os.getcwd()+"/safe_temp")

	if win!=None:
		win["info"].update("output ckpt file")
	else:
		print("output ckpt file")

	box=diff2sdxl()
	sd1=box.pipe(pipe.pipe)
	if base_path.endswith(".safetensors"):
		sd2=load_file(base_path)
	else:
		sd2=box.folder(base_path)
	flush()
	keys=zip_ckpt(sd1,sd2)
	save_safe(keys,out_path,os.getcwd()+"/safe_temp")

	if win!=None:
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")
	else:
		print("fin")

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading
	import pyperclip

	sg.theme('TealMono')

	keys=[
		'ckpt','vae','lora1','lora2','lora3',"out",'w1','w2','w3'
	]
	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]
	layout =[
		[sg.Text("checkpoint file")],
		[sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),sg.FileBrowse('select ckpt', file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("vae file")],
		[sg.InputText(key='vae',right_click_menu=grp_rclick_menu["vae"]),sg.FileBrowse('select vae', file_types=(('vae file', '.safetensors'),))],
		[sg.Text("lora1 file")],
		[sg.InputText(key='lora1',right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w1',right_click_menu=grp_rclick_menu["w1"])],
		[sg.Text("lora2 file")],
		[sg.InputText(key='lora2',right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w2',right_click_menu=grp_rclick_menu["w2"])],
		[sg.Text("lora3 file")],
		[sg.InputText(key='lora3',right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w3',right_click_menu=grp_rclick_menu["w3"])],
		[sg.Text("out file")],
		[sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]

	window = sg.Window('Make Safetensors Sdxl', layout)

	while True:
		event, values = window.read()
		if event in (sg.WIN_CLOSED, 'EXIT'):
			break

		elif event=="RUN":
			base_safe=values["ckpt"]
			loras=[]
			weights=[]
			out_safe=values["out"]
			vae_safe=values["vae"]
			if values["lora1"]!="":
				loras.append(values["lora1"])
				try:
					weights.append(float(values["w1"]))
					window["w1"].update(str(float(values["w1"])))
				except:
					weights.append(1.0)
					window["w1"].update("1.0")
					
			if values["lora2"]!="":
				loras.append(values["lora2"])
				try:
					weights.append(float(values["w2"]))
					window["w2"].update(str(float(values["w2"])))
				except:
					weights.append(1.0)
					window["w2"].update("1.0")
					
			if values["lora3"]!="":
				loras.append(values["lora3"])
				try:
					weights.append(float(values["w3"]))
					window["w3"].update(str(float(values["w3"])))
				except:
					weights.append(1.0)
					window["w3"].update("1.0")

			if base_safe!="" and out_safe!="":
				thread1 = threading.Thread(target=makesafe,args=(base_safe,loras,weights,out_safe,vae_safe,window))
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
				
	window.close()
