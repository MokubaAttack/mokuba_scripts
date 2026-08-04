import os
import shutil
import torch
from safetensors.torch import save_file
import math

from ..anima.mokuanipipe import mokuanipipe
from ..anima.animackpt2 import (
	safe2diff,
	folder2diff
)
from..common.diff2 import diff2anima
from ..common.save_safe import save_safe
from ..common.flush import flush

def zip_ckpt(pipe,transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,full_file):
	box=diff2anima()
	keys=[]
	for k,p in pipe.transformer.named_parameters():
		t=p.data.to(torch.float32)
		if k in transformer_sd:
			sum1=torch.sum(torch.abs(t)).item()
			sum2=torch.sum(torch.abs(transformer_sd.pop(k).to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t=t*sum2/sum1

		k=box.trans(k)

		osd={}
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	del transformer_sd

	for k,p in pipe.text_conditioner.named_parameters():
		t=p.data.to(torch.float32)
		if k in text_conditioner_sd:
			sum1=torch.sum(torch.abs(t)).item()
			sum2=torch.sum(torch.abs(text_conditioner_sd.pop(k).to(torch.float32))).item()
			n=not(math.isnan(sum1) or math.isnan(sum2))
			if n and sum1!=sum2:
				t=t*sum2/sum1

		k=box.teco(k)

		osd={}
		osd[k]=t.to(torch.bfloat16)
		save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
		keys.append(k)
	del text_conditioner_sd

	if full_file:
		for k,p in pipe.vae.named_parameters():
			t=p.data.to(torch.float32)
			if k in vae_sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(vae_sd.pop(k).to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1

			k=box.vae(k)

			osd={}
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
		del vae_sd
			
		for k,p in pipe.text_encoder.named_parameters():
			t=p.data.to(torch.float32)
			if k in text_encoder_sd:
				sum1=torch.sum(torch.abs(t)).item()
				sum2=torch.sum(torch.abs(text_encoder_sd.pop(k).to(torch.float32))).item()
				n=not(math.isnan(sum1) or math.isnan(sum2))
				if n and sum1!=sum2:
					t=t*sum2/sum1

			k=box.teen(k)

			osd={}
			osd[k]=t.to(torch.bfloat16)
			save_file(osd,os.getcwd()+"/safe_temp/"+k+".safetensors")
			keys.append(k)
		del text_encoder_sd
	else:
		del vae_sd,text_encoder_sd
			
	return keys

def makesafe(base_path,loras,ws,out_path,ff,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)
		
	if win!=None:
		win["info"].update("make pipe")
	else:
		print("make pipe")
	
	try:
		pipe=mokuanipipe()
		if base_path.endswith(".safetensors"):
			pipe.from_safe(base_path)
		else:
			pipe.from_folder(base_path)
	except:
		if win!=None:
			win["RUN"].Update(disabled=False)
			win["info"].update("Unsupported Anima checkpoint")
		else:
			print("Unsupported Anima checkpoint")
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
	if base_path.endswith(".safetensors"):
		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=safe2diff(path=base_path)
	else:
		transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd=folder2diff(path=base_path)
	keys=zip_ckpt(pipe.pipe,transformer_sd,text_conditioner_sd,text_encoder_sd,vae_sd,ff)
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
		'ckpt','lora1','lora2','lora3',"out",'w1','w2','w3'
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
		[
			sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),
		],
		[
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse", enable_events=True),
			sg.FolderBrowse("select ckpt folder",key="folder_browse", enable_events=True)
		],
		[sg.Text("lora1 file")],
		[sg.InputText(key='lora1',right_click_menu=grp_rclick_menu["lora1"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w1',right_click_menu=grp_rclick_menu["w1"])],
		[sg.Text("lora2 file")],
		[sg.InputText(key='lora2',right_click_menu=grp_rclick_menu["lora2"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w2',right_click_menu=grp_rclick_menu["w2"])],
		[sg.Text("lora3 file")],
		[sg.InputText(key='lora3',right_click_menu=grp_rclick_menu["lora3"]),sg.FileBrowse('select lora', file_types=(('lora file', '.safetensors'),))],
		[sg.Text("weight"),sg.InputText("1.0",key='w3',right_click_menu=grp_rclick_menu["w3"])],
		[sg.Checkbox('full file', key='ff')],
		[sg.Text("out file")],
		[sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]

	window = sg.Window('Make Safetensors Anima', layout)

	while True:
		event, values = window.read()
		if event in (sg.WIN_CLOSED, 'EXIT'):
			break

		elif event=="RUN":
			base_safe=values["ckpt"]
			loras=[]
			weights=[]
			out_safe=values["out"]
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
			full_file=values["ff"]

			if base_safe!="" and out_safe!="":
				thread1 = threading.Thread(target=makesafe,args=(base_safe,loras,weights,out_safe,full_file,window))
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
		elif event=="folder_browse":
			window["ckpt"].update(values["folder_browse"])
		elif event=="file_browse":
			window["ckpt"].update(values["file_browse"])
				
	window.close()
