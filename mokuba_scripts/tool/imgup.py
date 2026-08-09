import threading
import tkinter as tk
import pyperclip
import FreeSimpleGUI as sg
from PIL import Image
import torch

from ..common.upscaler import mokuup

def up_func(path,up_path,win):
	win["RUN"].update(disabled=True)
	if torch.cuda.is_available():
		d="cuda"
	elif torch.backends.mps.is_available():
		d="mps"
	elif torch.xpu.is_available():
		d="xpu"
	else:
		d="cpu"
	try:
		up=mokuup(up_path,d)
		s=up.get_scale()
		img=Image.open(path)
		x,y=img.size
		x=s*x
		y=s*y
		img=up.run(img,x,y)
		if path.endswith(".jpg"):
			out_path=path.replace(".jpg","_x"+str(s)+".jpg")
			img.save(out_path, 'JPEG' ,quality=85)
		else:
			out_path=path.replace(".png","_x"+str(s)+".png")
			img.save(out_path, "PNG")
		win["info"].update("fin")
		win["RUN"].update(disabled=False)
	except:
		win["info"].update("error")
		win["RUN"].update(disabled=False)

def gui():
	inters=["NEAREST","BOX","LANCZOS","HAMMING","BICUBIC","BILINEAR","select file"]
	keys=[
		'input_pic'
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
		[sg.Text("input file")],
		[sg.InputText(key='input_pic',right_click_menu=grp_rclick_menu["input_pic"]),sg.FileBrowse('input file', file_types=(('input file', '.jpg'),('input file', '.png'),))],
		[sg.Text("upscaler")],
		[sg.Combo(default_value="NEAREST",key="up",values=inters,enable_events = True)],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]
	window = sg.Window('imgup', layout)

	while True:
		event, values = window.read()
		if event in (sg.WIN_CLOSED, 'EXIT'):
			break

		elif event=="RUN":
			if values["input_pic"]!="":
				thread1 = threading.Thread(target=up_func,args=(values["input_pic"],values["up"],window))
				thread1.start()

		elif event=="up":
			if values["up"]=="select file":
				path = sg.popup_get_file('upscaler file',title="select upscaler file",file_types=(('upscaler file', '.pth'),))
				if path!=None:
					window["up"].update(path)
				else:
					window["up"].update("NEAREST")
					
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
