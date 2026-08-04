import torch
from safetensors.torch import (
	load_file,
	save_file
)

fp16_type = torch.float16
bf16_type = torch.bfloat16
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2

choices=[
	"fp8_e4m3","fp8_e5m2","fp16","bf16"
]

def conv(path,t,win=None):
	if win!=None:
		win['RUN'].Update(disabled=True)
	try:
		if t=="fp8_e4m3":
			a=e4m3_type
		elif t=="fp8_e5m2":
			a=e5m2_type
		elif t=="bf16":
			a=bf16_type
		else:
			a=fp16_type

		sd=load_file(path)
		for k,w in sd.items():
			sd[k]=w.to(a)
		out_path=path.replace(".safetensors","_"+t+".safetensors")
		save_file(sd,out_path)
		if win!=None:
			win["info"].update("fin")
			win['RUN'].Update(disabled=False)
		else:
			print("fin")
	except:
		if win!=None:
			win["info"].update("error")
			win['RUN'].Update(disabled=False)
		else:
			print("error")

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading
	import pyperclip

	sg.theme('TealMono')
	
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
		[
			sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),
			sg.FileBrowse('select file', file_types=(('safetensors file', '.safetensors'),),key="file_browse", enable_events=True),
		],
		[
				sg.Text("dtype"), sg.Combo(default_value="fp16",values=choices,key="dtype"),
		],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]
	
	window = sg.Window('Accuracy', layout)
	
	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["ckpt"]!="":
				thread1 = threading.Thread(target=conv,args=(values["ckpt"],values["dtype"],window))
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