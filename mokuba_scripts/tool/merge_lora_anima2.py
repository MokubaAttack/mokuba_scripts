import torch
import os
from safetensors.torch import save_file
import re

from ..common.diff2 import diff2anima
from ..common.dl import dlc
from ..common.icon import (
	green_icon,
	red_icon
)

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
	device=None,
	save_to=None,
	meta_dict=None,
	dof=False,
	win=None,
	token="",
):
	if win!=None:
		win.set_icon(pngbase64=red_icon)
		win['RUN'].Update(disabled=True)
	if len(loras) != len(weights):
		if win==None:
			print("number of models must be equal to number of ratios.")
		else:
			win["info"].update("number of models must be equal to number of ratios.")
			win.set_icon(pngbase64=green_icon)
			win['RUN'].Update(disabled=False)
		return

	merge_dtype = str_to_dtype(precision)
	save_dtype = str_to_dtype(save_precision)

	sds=[]
	keys=[]
	safe_folder=save_to.removesuffix(os.path.basename(save_to))
	box=diff2anima()
	for i in range(len(loras)):
		lora=str(loras[i])
		m=re.match(r"[0-9]+$",lora)
		if m!=None:
			ver_id=lora
			lora=safe_folder+lora+".safetensors"
			loras[i]=lora
			if token=="":
				if win==None:
					print("civitai token doesn't input.")
				else:
					win["info"].update("civitai token doesn't input.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			dlc(ver_id,lora,token)
		sd=box.lora(lora)
		if sd=={}:
			if win==None:
				print(os.path.basename(lora)+" isn't supported.")
			else:
				win["info"].update(os.path.basename(lora)+" isn't supported.")
				win.set_icon(pngbase64=green_icon)
				win['RUN'].Update(disabled=False)
			return
		sds.append(sd)
		keys=keys+list(sd)

	keys=list(set(keys))
	key_sum=len(keys)
	key_count=0
	merged_lora_sd={}

	if win==None:
		print("merge")

	for k in keys:
		key_count=key_count+1
		if win!=None:
			win["info"].update("merge : "+str(key_count)+"/"+str(key_sum))
		else:
			print("\r"+str(key_count)+"/"+str(key_sum),end="")
		if not(k.endswith(".lora_A.weight")):
			continue

		if k in sds[0] and k in sds[1]:
			wa=[0,0]
			wb=[0,0]
			wa[0]=sds[0].pop(k)
			wb[0]=sds[0].pop(k.replace(".lora_A.weight",".lora_B.weight"))
			if torch.any(torch.isnan(wa[0])) or torch.any(torch.isnan(wb[0])):
				if win==None:
					print(os.path.basename(loras[0])+" has nan.")
				else:
					win["info"].update(os.path.basename(loras[0])+" has nan.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			wa[1]=sds[1].pop(k)
			wb[1]=sds[1].pop(k.replace(".lora_A.weight",".lora_B.weight"))
			if torch.any(torch.isnan(wa[1])) or torch.any(torch.isnan(wb[1])):
				if win==None:
					print(os.path.basename(loras[1])+" has nan.")
				else:
					win["info"].update(os.path.basename(loras[1])+" has nan.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			if k.replace(".lora_A.weight",".alpha") in sds[0]:
				a=sds[0].pop(k.replace(".lora_A.weight",".alpha"))
				wa[0]=a/wa[0].size()[0]*wa[0]
				wb[0]=a/wa[0].size()[0]*wb[0]
			if k.replace(".lora_A.weight",".alpha") in sds[1]:
				a=sds[1].pop(k.replace(".lora_A.weight",".alpha"))
				wa[1]=a/wa[1].size()[0]*wa[1]
				wb[1]=a/wa[1].size()[0]*wb[1]
			network_dim = max([wa[0].size()[0],wa[1].size()[0]])

			if wa[0].size()[0]<network_dim:
				dummy=torch.zeros(network_dim-wa[0].size()[0],wa[0].size()[1])
				wa[0]=torch.cat((wa[0],dummy),0)
				dummy=torch.zeros(wb[0].size()[0],network_dim-wb[0].size()[1])
				wb[0]=torch.cat((wb[0],dummy),1)
			if wa[1].size()[0]<network_dim:
				dummy=torch.zeros(network_dim-wa[0].size()[0],wa[1].size()[1])
				wa[1]=torch.cat((wa[1],dummy),0)
				dummy=torch.zeros(wb[1].size()[0],network_dim-wb[1].size()[1])
				wb[1]=torch.cat((wb[1],dummy),1)
			wa=weights[0]*wa[0]+weights[1]*wa[1]
			wb=weights[0]*wb[0]+weights[1]*wb[1]
			alpha=torch.tensor(network_dim).to(save_dtype)
			
			if type(wa) == torch.Tensor and wa.dtype.is_floating_point and wa.dtype != save_dtype:
				wa=wa.to(save_dtype)
			if type(wb) == torch.Tensor and wb.dtype.is_floating_point and wb.dtype != save_dtype:
				wb=wb.to(save_dtype)
	
			merged_lora_sd[k.replace(".lora_A.weight",".lora_B.weight")] = wb.to("cpu").contiguous()
			merged_lora_sd[k] = wa.to("cpu").contiguous()
			merged_lora_sd[k.replace(".lora_A.weight",".alpha")]=alpha.to("cpu").contiguous()
		elif k in sds[0] and not(k in sds[1]):
			wa=sds[0].pop(k)
			wb=sds[0].pop(k.replace(".lora_A.weight",".lora_B.weight"))
			if torch.any(torch.isnan(wa)) or torch.any(torch.isnan(wb)):
				if win==None:
					print(os.path.basename(loras[0])+" has nan.")
				else:
					win["info"].update(os.path.basename(loras[0])+" has nan.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			network_dim = wa.size()[0]
			alpha=sds[0].pop(k.replace(".lora_A.weight",".alpha"),network_dim)
			if type(alpha) == torch.Tensor:
				alpha=alpha.to(save_dtype)
			else:
				alpha=torch.tensor(alpha).to(save_dtype)

			if type(wa) == torch.Tensor and wa.dtype.is_floating_point and wa.dtype != save_dtype:
				wa=wa.to(save_dtype)
			if type(wb) == torch.Tensor and wb.dtype.is_floating_point and wb.dtype != save_dtype:
				wb=wb.to(save_dtype)
	
			merged_lora_sd[k.replace(".lora_A.weight",".lora_B.weight")] = weights[0]*wb.to("cpu").contiguous()
			merged_lora_sd[k] = weights[0]*wa.to("cpu").contiguous()
			merged_lora_sd[k.replace(".lora_A.weight",".alpha")]=alpha.to("cpu").contiguous()
		elif not(k in sds[0]) and k in sds[1]:
			wa=sds[1].pop(k)
			wb=sds[1].pop(k.replace(".lora_A.weight",".lora_B.weight"))
			if torch.any(torch.isnan(wa)) or torch.any(torch.isnan(wb)):
				if win==None:
					print(os.path.basename(loras[1])+" has nan.")
				else:
					win["info"].update(os.path.basename(loras[1])+" has nan.")
					win.set_icon(pngbase64=green_icon)
					win['RUN'].Update(disabled=False)
				return
			network_dim = wa.size()[0]
			alpha=sds[1].pop(k.replace(".lora_A.weight",".alpha"),network_dim)
			if type(alpha) == torch.Tensor:
				alpha=alpha.to(save_dtype)
			else:
				alpha=torch.tensor(alpha).to(save_dtype)

			if type(wa) == torch.Tensor and wa.dtype.is_floating_point and wa.dtype != save_dtype:
				wa=wa.to(save_dtype)
			if type(wb) == torch.Tensor and wb.dtype.is_floating_point and wb.dtype != save_dtype:
				wb=wb.to(save_dtype)
	
			merged_lora_sd[k.replace(".lora_A.weight",".lora_B.weight")] = weights[1]*wb.to("cpu").contiguous()
			merged_lora_sd[k] = weights[1]*wa.to("cpu").contiguous()
			merged_lora_sd[k.replace(".lora_A.weight",".alpha")]=alpha.to("cpu").contiguous()

	if isinstance(meta_dict, dict):
		save_file(merged_lora_sd,save_to,metadata=meta_dict)
	else:
		save_file(merged_lora_sd,save_to)
	if dof:
		for lora in loras:
			os.remove(lora)
			os.remove(lora+".txt")
	if win==None:
		print("")
		print("fin")
	else:
		win["info"].update("fin")
		win.set_icon(pngbase64=green_icon)
		win['RUN'].Update(disabled=False)

def gui():
	import threading
	import tkinter as tk
	import pyperclip
	import FreeSimpleGUI as sg
	import json

	from ..tool.civitai_dl import check

	if check()==False:
		return

	grp_rclick_menu={}
	keys=["ckpt1","ckpt2","w1","w2","id1","id2","out"]
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
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('lora file', '.safetensors'),)),sg.Button('clear', key='clear_out')],
		[sg.Checkbox('del original files', key='dof')],
		[sg.Text("information",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Merge Lora Anima2', layout,keep_on_top=True,icon=green_icon)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			c1=values["ckpt1"]!="" and values["ckpt2"]!=""
			if c1:
				names=[]
				weights=[]
				ids=[]
				for i in range(2):
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

				meta={}
				meta["id"]=str(ids).replace("[","").replace("]","").replace(" ","")
				meta["weight"]=str(weights).replace("[","").replace("]","").replace(" ","")

				try:
					with open(os.getcwd()+"/pipecache/token.json","r") as f:
						d = json.load(f)
					token=d["civitai_token"]
				except:
					token=""

				ok = sg.popup_ok_cancel(out_path,title='output file',keep_on_top=True)
				if ok=="OK":
					thread1 = threading.Thread(target=mergelora,args=(names,weights,"float","bf16",None,out_path,meta,values["dof"],window,token))
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
