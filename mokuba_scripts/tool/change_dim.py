from safetensors.torch import (
	save_file,
	load_file
)
import torch
import safetensors

CLAMP_QUANTILE=0.99

def str_to_dtype(p):
	if p == "bf16":
		return torch.bfloat16
	elif p == "fp16":
		return torch.float16
	else:
		return torch.float

def changedim(
	path="",
	precision="float",
	save_precision="fp16",
	new_rank=16,
	new_conv_rank=None,
	device=None,
	win=None
):
	if win!=None:
		win["RUN"].update(disabled=True)
		win["info"].update("changing")
	merge_dtype = str_to_dtype(precision)
	save_dtype = str_to_dtype(save_precision)
	
	new_conv_rank = new_conv_rank if new_conv_rank is not None else new_rank
	save_to=path.replace(".safetensors","_cha.safetensors")
	try:
		f=safetensors.safe_open(p, framework="pt", device="cpu")
		meta_dict=f.metadata()
	except:
		meta_dict={"format":"pt"}
	try:
		s=load_file(path)
	except:
		if win!=None:
			win["RUN"].update(disabled=False)
			win["info"].update("error : I couldn't load lora.")
		else:
			print("error : I couldn't load lora.")
		return
			
	keys=list(s)
	for k in keys:
		if k.endswith(".lora_A.weight"):
			k2=k.replace(".lora_A.weight",".lora_down.weight")
			s[k2]=s[k]
			del s[k]
		elif k.endswith(".lora_B.weight"):
			k2=k.replace(".lora_B.weight",".lora_up.weight")
			s[k2]=s[k]
			del s[k]
		
	merged_lora_sd={}
	keys=list(s)
	for k in keys:
		if not(k.endswith(".lora_down.weight")):
			continue
		try:
			wa=s.pop(k)
			wb=s.pop(k.replace(".lora_down.weight",".lora_up.weight"))
		except:
			if win!=None:
				win["RUN"].update(disabled=False)
				win["info"].update("error : lora dosen't be supported.")
			else:
				print("error : lora dosen't be supported.")
			return
		
		network_dim = wa.size()[0]
		alpha=s.pop(k.replace(".lora_down.weight",".alpha"),network_dim)
		in_dim = wa.size()[1]
		out_dim = wb.size()[0]
		conv2d = len(wa.size()) == 4
		kernel_size = None if not conv2d else wa.size()[2:4]
		scale = alpha / network_dim
		
		mat = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
		if not conv2d:
			mat = mat + (wb @ wa) * scale
		elif kernel_size == (1, 1):
			mat = (
				mat
				+ (wb.squeeze(3).squeeze(2) @ wa.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
				* scale
			)
		else:
			conved = torch.nn.functional.conv2d(wa.permute(1, 0, 2, 3), wb).permute(1, 0, 2, 3)
			mat = mat + conved * scale
			
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
	
	save_file(merged_lora_sd,save_to,metadata=meta_dict)
	win["RUN"].update(disabled=False)
	win["info"].update("fin")

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import threading
	import pyperclip

	sg.theme('TealMono')
	
	choices=[
		"fp32","fp16","bf16"
	]
	
	keys=['ckpt',"dim"]
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
			sg.Text("lora"),
			sg.InputText(key='ckpt',right_click_menu=grp_rclick_menu["ckpt"]),
			sg.FileBrowse('select file', file_types=(('safetensors file', '.safetensors'),),key="file_browse", enable_events=True),
		],
		[
				sg.Text("dim"), sg.InputText(key='dim',right_click_menu=grp_rclick_menu["dim"]),
		],
		[
				sg.Text("dtype"), sg.Combo(default_value="fp16",values=choices,key="dtype"),
		],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN'),sg.Button('EXIT')]
	]
	
	window = sg.Window('Change Dim', layout)
	
	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["ckpt"]!="" and values["dim"]!="":
				try:
					dim=int(values["dim"])
					if dim<=0:
						dim=8
				except:
					dim=8
				thread1 = threading.Thread(target=changedim,args=(values["ckpt"],"fp32",values["dtype"],dim,None,None,window))
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