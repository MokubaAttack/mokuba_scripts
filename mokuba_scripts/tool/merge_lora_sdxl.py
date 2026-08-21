import torch
import os
from safetensors.torch import save_file
import re

from ..common.diff2 import diff2sdxl
from ..common.dl import dlc

CLAMP_QUANTILE=0.99

def str_to_dtype(p):
	if p == "bf16":
		return torch.bfloat16
	elif p == "fp16":
		return torch.float16
	else:
		return torch.float

def mergelora(
	loras=[],
	weights=[],
	precision="float",
	save_precision="fp16",
	new_rank=16,
	new_conv_rank=None,
	device=None,
	save_to=None,
	meta_dict=None,
	dof=False,
	win=None,
	token="",
):
	if win!=None:
		win['RUN'].Update(disabled=True)
	if len(loras) != len(weights):
		if win==None:
			print("number of models must be equal to number of ratios.")
		else:
			win["info"].update("number of models must be equal to number of ratios.")
			win['RUN'].Update(disabled=False)
		return

	merge_dtype = str_to_dtype(precision)
	save_dtype = str_to_dtype(save_precision)

	new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank
	sds=[]
	keys=[]
	safe_folder=save_to.removesuffix(os.path.basename(save_to))
	box=diff2sdxl()
	for i in range(len(loras)):
		lora=str(loras[i])
		m=re.match(r"[0-9]+$",lora)
		if m!=None:
			ver_id=lora
			lora=safe_folder+lora+".safetensors"
			loras[i]=loras
			if token=="":
				if win==None:
					print("civitai token doesn't input.")
				else:
					win["info"].update("civitai token doesn't input.")
					win['RUN'].Update(disabled=False)
				return
			dlc(ver_id,lora,token)
		sd=box.lora(lora)
		if sd=={}:
			if win==None:
				print(os.path.basename(lora)+" isn't supported.")
			else:
				win["info"].update(os.path.basename(lora)+" isn't supported.")
				win['RUN'].Update(disabled=False)
			return
		sds.append(sd)
		keys=keys+list(sd)

	keys=list(set(keys))
	key_sum=len(keys)
	key_count=0
	merged_lora_sd={}

	if win==None:
		print("svd")

	for k in keys:
		key_count=key_count+1
		if win!=None:
			win["info"].update("svd : "+str(key_count)+"/"+str(key_sum))
		else:
			print("\r"+str(key_count)+"/"+str(key_sum),end="")
		if not(k.endswith(".lora_down.weight")):
			continue

		first_take=True
		for i in range(len(sds)):
			if not(k in sds[i]):
				continue
			wa=sds[i].pop(k)
			wb=sds[i].pop(k.replace(".lora_down.weight",".lora_up.weight"))
			if torch.any(torch.isnan(wa)) or torch.any(torch.isnan(wb)):
				if win==None:
					print(os.path.basename(loras[i])+" has nan.")
				else:
					win["info"].update(os.path.basename(loras[i])+" has nan.")
					win['RUN'].Update(disabled=False)
				return

			network_dim = wa.size()[0]
			alpha=sds[i].pop(k.replace(".lora_down.weight",".alpha"),network_dim)
			in_dim = wa.size()[1]
			out_dim = wb.size()[0]
			conv2d = len(wa.size()) == 4
			kernel_size = None if not conv2d else wa.size()[2:4]
			scale = alpha / network_dim

			if first_take:
				mat = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
				first_take=False

			if device:
				mat = mat.to(device)
				wb = wb.to(device)
				wa = wa.to(device)
				scale = scale.to(device)
				
			if not conv2d:
				mat = mat + weights[i] * (wb @ wa) * scale
			elif kernel_size == (1, 1):
				mat = (
					mat
					+ weights[i]
					* (wb.squeeze(3).squeeze(2) @ wa.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
					* scale
				)
			else:
				conved = torch.nn.functional.conv2d(wa.permute(1, 0, 2, 3), wb).permute(1, 0, 2, 3)
				mat = mat + weights[i] * conved * scale

		conv2d = len(mat.size()) == 4
		kernel_size = None if not conv2d else mat.size()[2:4]
		conv2d_3x3 = conv2d and kernel_size != (1, 1)
		out_dim, in_dim = mat.size()[0:2]

		if conv2d:
			if conv2d_3x3:
				mat = mat.flatten(start_dim=1)
			else:
				mat = mat.squeeze()

		module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
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
		a=torch.tensor(module_new_rank).to(save_dtype)

		if type(up_weight) == torch.Tensor and up_weight.dtype.is_floating_point and up_weight.dtype != save_dtype:
			up_weight=up_weight.to(save_dtype)
		if type(down_weight) == torch.Tensor and down_weight.dtype.is_floating_point and down_weight.dtype != save_dtype:
			down_weight=down_weight.to(save_dtype)

		merged_lora_sd[k.replace(".lora_down.weight",".lora_up.weight")] = up_weight.to("cpu").contiguous()
		merged_lora_sd[k] = down_weight.to("cpu").contiguous()
		merged_lora_sd[k.replace(".lora_down.weight",".alpha")]=a.to("cpu").contiguous()

	if isinstance(meta_dict, dict):
		save_file(merged_lora_sd,save_to,metadata=meta_dict)
	else:
		save_file(merged_lora_sd,save_to)
	if dof:
		for lora in loras:
			os.remove(lora)
	if win==None:
		print("")
		print("fin")
	else:
		win["info"].update("fin")
		win['RUN'].Update(disabled=False)

def gui():
	import threading
	import tkinter as tk
	import pyperclip
	import FreeSimpleGUI as sg
	import json

	from ..tool.civitai_dl import check

	sg.theme('TealMono')

	if check()==False:
		return

	grp_rclick_menu={}
	keys=["ckpt1","ckpt2","ckpt3","ckpt4","w1","w2","w3","w4","id1","id2","id3","id4","out","d"]
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]		
			
	layout=[
		[sg.Text("lora1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear1')],
		[sg.Text("weight"),sg.Input("0.7",key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id1",right_click_menu=grp_rclick_menu["id1"], size=(20, 1))],
		[sg.Text("lora2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear2')],
		[sg.Text("weight"),sg.Input("0.7",key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id2",right_click_menu=grp_rclick_menu["id2"], size=(20, 1))],
		[sg.Text("lora3"), sg.Input(key="ckpt3",right_click_menu=grp_rclick_menu["ckpt3"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear3')],
		[sg.Text("weight"),sg.Input("0.7",key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id3",right_click_menu=grp_rclick_menu["id3"], size=(20, 1))],
		[sg.Text("lora4"), sg.Input(key="ckpt4",right_click_menu=grp_rclick_menu["ckpt4"]),sg.FileBrowse( file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear4')],
		[sg.Text("weight"),sg.Input("0.7",key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1)),sg.Text("id"),sg.Input("",key="id4",right_click_menu=grp_rclick_menu["id4"], size=(20, 1))],
		[sg.Text("dim"), sg.Input("16",key="d",right_click_menu=grp_rclick_menu["d"])],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear_out')],
		[sg.Checkbox('del original files', key='dof')],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Merge Lora Sdxl', layout,keep_on_top=True)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			c1=not(values["ckpt1"]=="" and values["ckpt2"]=="" and values["ckpt3"]=="" and values["ckpt4"]=="")
			if c1:
				names=[]
				weights=[]
				ids=[]
				for i in range(4):
					if values["ckpt"+str(i+1)]!="":
						names.append(values["ckpt"+str(i+1)])
						try:
							weights.append(float(values["w"+str(i+1)]))
						except:
							weights.append(0.7)
						window["w"+str(i+1)].update(str(weights[-1]))
						try:
							ids.append(int(values["id"+str(i+1)]))
						except:
							ids.append(5)
						window["id"+str(i+1)].update(str(ids[-1]))
				if values["out"]=="":
					out_path=""
					for line in names:
						if out_path=="":
							out_path=os.path.dirname(line)+"/"+os.path.basename(line).split(".")[0]
						else:
							out_path=out_path+"_"+os.path.basename(line).split(".")[0]
					out_path=out_path+".safetensors"
					window["out"].update(out_path)
				else:
					out_path=values["out"]
					if not(out_path.endswith(".safetensors")):
						out_path=out_path+".safetensors"
						window["out"].update(out_path)
				
				try:
					dim=int(values["d"])
				except:
					dim=16
				window["d"].update(str(dim))

				meta={}
				meta["id"]=str(ids).replace("[","").replace("]","").replace(" ","")
				meta["weight"]=str(weights).replace("[","").replace("]","").replace(" ","")

				try:
					with open(os.getcwd()+"/token.json","r") as f:
						d = json.load(f)
					token=d["civitai_token"]
				except:
					token=""

				ok = sg.popup_ok_cancel(out_path,title='output file',keep_on_top=True)
				if ok=="OK":
					thread1 = threading.Thread(target=mergelora,args=(names,weights,"float","bf16",dim,None,None,out_path,meta,values["dof"],window,token))
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
		elif "clear" in event:
			try:
				if event=="clear_out":
					key="out"
				else:
					key="ckpt"+event.replace("clear","")
					key2="id"+event.replace("clear","")
					window[key2].update("")
				window[key].update("")
			except:
				pass

	window.close()
