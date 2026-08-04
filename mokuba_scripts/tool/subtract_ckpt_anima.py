import os
import torch
from safetensors.torch import save_file

from ..anima.animackpt2 import (
	safe2diff,
	folder2diff
)
from ..common.diff2 import diff2anima
from ..common.flush import flush

CLAMP_QUANTILE=0.99

def subtractckpt(ckpts,dim,trans,teco,teen,out_path,win=None):
	if win!=None:
		win["RUN"].Update(disabled=True)

	for path in ckpts:
		if not(os.path.exists(path)):
			if win==None:
				print(path+" does not exist.")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update(path+" does not exist.")
			return

	if win!=None:
		win["info"].update("loading "+os.path.basename(ckpts[0]))
	else:
		print("loading "+os.path.basename(ckpts[0]))
	if ckpts[0].endswith(".safetensors"):
		try:
			sd10,sd11,sd12,sd13=safe2diff(path=ckpts[0],trans=trans,teco=teco,teen=teen,vae=False)
		except:
			if win==None:
				print("I couldn't load "+os.path.basename(ckpts[0])+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+os.path.basename(ckpts[0])+".")
			return
	else:
		try:
			sd10,sd11,sd12,sd13=folder2diff(path=ckpts[0],trans=trans,teco=teco,teen=teen,vae=False)
		except:
			if win==None:
				print("I couldn't load "+ckpts[0].replace("\\","/").split("/")[-1]+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+ckpts[0].replace("\\","/").split("/")[-1]+".")
			return

	if ckpts[1].endswith(".safetensors"):
		try:
			sd20,sd21,sd22,sd23=safe2diff(path=ckpts[1],trans=trans,teco=teco,teen=teen,vae=False)
		except:
			if win==None:
				print("I couldn't load "+os.path.basename(ckpts[1])+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+os.path.basename(ckpts[1])+".")
			return
	else:
		try:
			sd20,sd21,sd22,sd23=folder2diff(path=ckpts[1],trans=trans,teco=teco,teen=teen,vae=False)
		except:
			if win==None:
				print("I couldn't load "+ckpts[1].replace("\\","/").split("/")[-1]+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+ckpts[1].replace("\\","/").split("/")[-1]+".")
			return

	keys0=list(sd10)
	keys1=list(sd11)
	keys2=list(sd12)
	keys3=list(sd13)
	dict_sum=len(keys0+keys1+keys2+keys3)
	key_count=0

	box=diff2anima()
	out_dict={}
	for k in keys0:
		key_count=key_count+1
		if win!=None:
			win["info"].update("subtracting "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rsubtracting "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd10.pop(k).to(torch.float32)
			t2=sd20.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			return
			
		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue
		
		key=box.trans(k).removeprefix("model.diffusion_model.").removesuffix(".weight").replace(".","_")
		
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

		out_dict["lora_unet_"+key + ".lora_up.weight"] = up_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_unet_"+key + ".lora_down.weight"] = down_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_unet_"+key + ".alpha"] = a.to(torch.bfloat16)
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val,a
		flush()
	del sd10,sd20
		
	for k in keys1:
		key_count=key_count+1
		if win!=None:
			win["info"].update("subtracting "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rsubtracting "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd11.pop(k).to(torch.float32)
			t2=sd21.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			return
			
		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue
		
		key=box.teco(k).removeprefix("model.diffusion_model.").removesuffix(".weight").replace(".","_")
		if not(key.startswith("llm_adapter_blocks")):
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

		out_dict["lora_unet_"+key + ".lora_up.weight"] = up_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_unet_"+key + ".lora_down.weight"] = down_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_unet_"+key + ".alpha"] = a.to(torch.bfloat16)
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val,a
		flush()
	del sd11,sd21
		
	for k in keys2:
		key_count=key_count+1
		if win!=None:
			win["info"].update("subtracting "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rsubtracting "+str(key_count)+"/"+str(dict_sum),end="")

		try:
			t1=sd12.pop(k).to(torch.float32)
			t2=sd22.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			return
			
		mat=t2-t1

		if mat.dim()==1:
			continue
		if torch.any(torch.isnan(mat)):
			continue
		
		key=box.teen(k).removeprefix("cond_stage_model.qwen3_06b.transformer.model.").removesuffix(".weight").replace(".","_")
		if not(key.startswith("layers")):
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

		out_dict["lora_te_"+key + ".lora_up.weight"] = up_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_te_"+key + ".lora_down.weight"] = down_weight.to(torch.bfloat16).contiguous()
		out_dict["lora_te_"+key + ".alpha"] = a.to(torch.bfloat16)
		del t1,t2,mat,up_weight,down_weight,U,S,Vh,dist,hi_val,low_val,a
		flush()
	del sd12,sd22
	
	save_file(out_dict,out_path)
	if win==None:
		print("")
		print("fin")
	else:
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")
	
def gui():
	import tkinter as tk
	import pyperclip
	import threading
	import FreeSimpleGUI as sg

	sg.theme('TealMono')

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
			sg.FolderBrowse("select ckpt folder",key="folder_browse1", enable_events=True)
		],
		[
			sg.Text("model_tuned"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),
			sg.FileBrowse('select ckpt file', file_types=(('ckpt file', '.safetensors'),),key="file_browse2", enable_events=True),
			sg.FolderBrowse("select ckpt folder",key="folder_browse2", enable_events=True)
		],
		[sg.Text("dim"), sg.Input(key="dim",right_click_menu=grp_rclick_menu["dim"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),))],
		[sg.Checkbox('transformer', default=True, key='unet'),sg.Checkbox('text_conditioner', default=True, key='text2'),sg.Checkbox('text_encoder', default=True, key='text1')],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Subtract Ckpt Anima', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
				if values["dim"]=="":
					dim=8
				else:
					try:
						dim=abs(int(values["dim"]))
					except:
						dim=8
				window["dim"].update(str(dim))
				out_path=values["out"]
				paths=[
					values["ckpt1"],values["ckpt2"]
				]

				trans_out=values["unet"]
				teen_out=values["text1"]
				teco_out=values["text2"]
				
				thread1 = threading.Thread(target=subtractckpt,args=(paths,dim,trans_out,teco_out,teen_out,out_path,window))
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
		elif event=="folder_browse1":
			window["ckpt1"].update(values["folder_browse1"])
		elif event=="file_browse2":
			window["ckpt2"].update(values["file_browse2"])
		elif event=="folder_browse2":
			window["ckpt2"].update(values["folder_browse2"])
		
	window.close()
	