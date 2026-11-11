import torch
from safetensors.torch import (
	load_file,
	save_file
)

from ..common.icon import (
	green_icon,
	red_icon
)
	
def get_vae(path,dtype,win=None):
	if win!=None:
		win.set_icon(pngbase64=red_icon)
		win["RUN"].Update(disabled=True)
	if path.endswith(".safetensors"):
		if dtype=="bf16":
			dtype=torch.bfloat16
		elif dtype=="fp16":
			dtype=torch.float16
		else:
			dtype=torch.float32
		key="first_stage_model."
		base_safe=path
		out=base_safe.replace(".safetensors","_vae.safetensors")
		try:
			state_dict=load_file(base_safe)
		except:
			if win==None:
				print("load error")
			else:
				win.set_icon(pngbase64=green_icon)
				win["info"].update("load error")
				win["RUN"].Update(disabled=False)
			return
		out_dict={}
		for k,w in state_dict.items():
			if k.startswith(key):
				k_out=k.replace(key,"")
				out_dict[k_out]=w.to(dtype)
		try:
			save_file(out_dict,out)
		except:
			if win==None:
				print("save error")
			else:
				win.set_icon(pngbase64=green_icon)
				win["info"].update("save error")
				win["RUN"].Update(disabled=False)
			return
		if win==None:
			print("fin")
		else:
			win.set_icon(pngbase64=green_icon)
			win["info"].update("fin")
			win["RUN"].Update(disabled=False)
	else:
		if win==None:
			print("support safetensors file only")
		else:
			win.set_icon(pngbase64=green_icon)
			win["info"].update("support safetensors file only")
			win["RUN"].Update(disabled=False)

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading
	import pyperclip
	
	keys=['ckpt']
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
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse", enable_events=True),
		],
		[
				sg.Text("dtype"), sg.Combo(default_value="f16",values=["fp32","fp16","bf16"],key="dtype"),
		],
		[sg.Text("information",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]
	
	window = sg.Window('Get Vae', layout,icon=green_icon)
	
	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["ckpt"]!="":
				thread1 = threading.Thread(target=get_vae,args=(values["ckpt"],values["dtype"],window))
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
		elif event=="file_browse":
			window["input"].update(values["file_browse"])
	
	window.close()
