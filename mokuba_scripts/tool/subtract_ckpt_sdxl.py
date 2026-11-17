import torch
import os
from safetensors.torch import (
	save_file,
	load_file
)
import re

from ..common.diff2 import diff2sdxl
from ..common.flush import flush
from ..common.dl import dlc
from ..common.icon import (
	green_icon,
	red_icon
)

CLAMP_QUANTILE=0.99

def subfunc(path,i):
	box=diff2sdxl()
	sd1=box.pipelora(path)
	for k in sd1:
		sd1[k]=sd1[k].to(torch.float16)
	out_path="pipecache/"+str(i)+".safetensors"
	save_file(sd1,out_path)
	return list(sd1),out_path

def subtractckpt(ckpts,dim,trans,teen1,teen2,out_path,win=None,token=""):
	if win!=None:
		win.set_icon(pngbase64=red_icon)
		win["RUN"].Update(disabled=True)

	if not(out_path.endswith(".safetensors")):
		if win==None:
			print("the output path is needed to be a safetensors file.")
		else:
			win.set_icon(pngbase64=green_icon)
			win["RUN"].Update(disabled=False)
			win["info"].update("the output path is needed to be a safetensors file.")
		return

	safe_folder=out_path.removesuffix(os.path.basename(out_path))
	for ind,path in enumerate(ckpts):
		path=str(path)
		m=re.match(r"[0-9]+$",path)
		if m!=None:
			ver_id=path
			path=safe_folder+path+".safetensors"
			ckpts[ind]=path
			if token=="":
				if win==None:
					print("civitai token doesn't input.")
				else:
					win["info"].update("civitai token doesn't input.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			dlc(ver_id,path,token)
		if not(os.path.exists(path)):
			if win==None:
				print(path+" does not exist.")
			else:
				win.set_icon(pngbase64=green_icon)
				win["RUN"].Update(disabled=False)
				win["info"].update(os.path.basename(path)+" does not exist.")
			return
	
	if win!=None:
		win["info"].update("loading "+os.path.basename(ckpts[0]))
	else:
		print("loading "+os.path.basename(ckpts[0]))

	try:
		keys0,p0=subfunc(ckpts[0],0)
	except:
		if win!=None:
			win["info"].update(os.path.basename(ckpts[0])+" isn't supported.")
			win.set_icon(pngbase64=green_icon)
			win["RUN"].Update(disabled=False)
		else:
			print(os.path.basename(ckpts[0])+" isn't supported.")
		return
		
	if win!=None:
		win["info"].update("loading "+os.path.basename(ckpts[1]))
	else:
		print("loading "+os.path.basename(ckpts[1]))
		
	try:
		_,p1=subfunc(ckpts[1],1)
	except:
		if win!=None:
			win["info"].update(os.path.basename(ckpts[1])+" isn't supported.")
			win.set_icon(pngbase64=green_icon)
			win["RUN"].Update(disabled=False)
		else:
			print(os.path.basename(ckpts[1])+" isn't supported.")
		return
		
	sd1=load_file(p0)
	sd2=load_file(p1)
	dict_sum=len(keys0)
	key_count=0
	out_dict={}
	
	for k in keys0:
		key_count=key_count+1
		if win!=None:
			win["info"].update("subtracting "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rsubtracting "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd1.pop(k).to(torch.float32)
			t2=sd2.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Sdxl checkpoint key: "+k)
			else:
				win.set_icon(pngbase64=green_icon)
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Sdxl checkpoint key: "+k)
			return
			
		mat=t2-t1

		if torch.equal(t1,t2):
			continue
		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue
		
		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = dim
		module_new_rank = min(module_new_rank, in_dim, out_dim)

		U, S, Vh = torch.linalg.svd(mat)

		U = U[:, :module_new_rank]
		S = S[:module_new_rank]
		U = U @ torch.diag(S)

		Vh = Vh[:module_new_rank, :]

		dist = torch.cat([U.flatten(), Vh.flatten()])
		hi_val = torch.quantile(dist, CLAMP_QUANTILE)
		low_val = -hi_val

		U = U.clamp(low_val, hi_val)
		Vh = Vh.clamp(low_val, hi_val)

		if conv2d:
			U = U.reshape(out_dim, module_new_rank, 1, 1)
			Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

		up_weight = U
		down_weight = Vh
		a=torch.tensor(module_new_rank)

		out_dict[k + ".lora_up.weight"] = up_weight.to(torch.float16).contiguous()
		out_dict[k + ".lora_down.weight"] = down_weight.to(torch.float16).contiguous()
		out_dict[k + ".alpha"] = a.to(torch.float16)
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val,a
		flush()
	del sd1,sd2
	os.remove(p1)
	os.remove(p0)
	save_file(out_dict,out_path)
	if win==None:
		print("")
		print("fin")
	else:
		win.set_icon(pngbase64=green_icon)
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")
	
def gui():
	import tkinter as tk
	import pyperclip
	import threading
	import FreeSimpleGUI as sg
	import json

	from ..tool.civitai_dl import check

	if check()==False:
		return

	keys=["ckpt1","ckpt2","out","dim"]

	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]

	layout=[
		[
			sg.Text("model_org"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse1", enable_events=True),
		],
		[
			sg.Text("model_tuned"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse2", enable_events=True),
		],
		[sg.Text("dim"), sg.Input(key="dim",right_click_menu=grp_rclick_menu["dim"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),))],
		[sg.Checkbox('unet', default=True, key='unet'),sg.Checkbox('text_encoder', default=True, key='text1'),sg.Checkbox('text_encoder_2', default=True, key='text2')],
		[sg.Text("information",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Subtract Ckpt Sdxl', layout,icon=green_icon)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
				if values["dim"]=="":
					dim=16
				else:
					try:
						dim=abs(int(values["dim"]))
					except:
						dim=16
				window["dim"].update(str(dim))
				out_path=values["out"]
				paths=[
					values["ckpt1"],values["ckpt2"]
				]

				trans_out=values["unet"]
				teen1_out=values["text1"]
				teen2_out=values["text2"]

				try:
					with open(os.getcwd()+"/pipecache/token.json","r") as f:
						d = json.load(f)
					token=d["civitai_token"]
				except:
					token=""
				
				thread1 = threading.Thread(target=subtractckpt,args=(paths,dim,trans_out,teen1_out,teen2_out,out_path,window,token))
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
		elif event=="file_browse1":
			window["ckpt1"].update(values["file_browse1"])
		elif event=="file_browse2":
			window["ckpt2"].update(values["file_browse2"])
		
	window.close()
	
