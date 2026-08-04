import os
import shutil
from safetensors.torch import save_file
import torch

from ..anima.animackpt2 import (
	safe2diff,
	folder2diff
)
from ..common.diff2 import diff2anima
from ..common.flush import flush
from ..common.save_safe import save_safe

def mergeckpt(ckpts,ws,out_path,mode="normal",ff=True,win=None,v=0):
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

	if not(mode in ["normal","tensor1","tensor2"]):
		mode="normal"

	if os.path.exists(os.getcwd()+"/safe_temp"):
		shutil.rmtree(os.getcwd()+"/safe_temp")
	os.mkdir(os.getcwd()+"/safe_temp")

	if ff:
		ff=(True,True,True,True)
	else:
		ff=(True,True,False,False)

	if win!=None:
		win["info"].update("loading "+os.path.basename(ckpts[0]))
	else:
		print("loading "+os.path.basename(ckpts[0]))
	if ckpts[0].endswith(".safetensors"):
		try:
			sd10,sd11,sd12,sd13=safe2diff(path=ckpts[0],trans=ff[0],teco=ff[1],teen=ff[2],vae=ff[3])
		except:
			if win==None:
				print("I couldn't load "+os.path.basename(ckpts[0])+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+os.path.basename(ckpts[0])+".")
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return
	else:
		try:
			sd10,sd11,sd12,sd13=folder2diff(path=ckpts[0],trans=ff[0],teco=ff[1],teen=ff[2],vae=ff[3])
		except:
			if win==None:
				print("I couldn't load "+ckpts[0].replace("\\","/").split("/")[-1]+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+ckpts[0].replace("\\","/").split("/")[-1]+".")
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

	if ckpts[1].endswith(".safetensors"):
		try:
			sd20,sd21,sd22,sd23=safe2diff(path=ckpts[1],trans=ff[0],teco=ff[1],teen=ff[2],vae=ff[3])
		except:
			if win==None:
				print("I couldn't load "+os.path.basename(ckpts[1])+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+os.path.basename(ckpts[1])+".")
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return
	else:
		try:
			sd20,sd21,sd22,sd23=folder2diff(path=ckpts[1],trans=ff[0],teco=ff[1],teen=ff[2],vae=ff[3])
		except:
			if win==None:
				print("I couldn't load "+ckpts[1].replace("\\","/").split("/")[-1]+".")
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("I couldn't load "+ckpts[1].replace("\\","/").split("/")[-1]+".")
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

	keys0=list(sd10)
	keys1=list(sd11)
	keys2=list(sd12)
	keys3=list(sd13)
	data_dict=[]
	dict_sum=len(keys0+keys1+keys2+keys3)
	key_count=0

	box=diff2anima()

	for k in keys0:
		key_count=key_count+1
		if win!=None:
			win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")

		out_dict={}
		try:
			t1=sd10.pop(k).to(torch.float32)
			t2=sd20.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

		k=box.trans(k)

		if k.startswith("model.diffusion_model.blocks."):
			ind=int(k.split(".")[3])
			w=ws[ind+1]
		else:
			w=ws[0]

		if mode=="normal":
			out_dict[k]=((1-w)*t1+w*t2).to(torch.bfloat16)

		elif "tensor" in mode:
			w1=(1-w)/2
			w2=w
			w1=round(t1.size()[0]*w1)
			w2=round(t1.size()[0]*(w1+w2))
			if w1==0:
				out_dict[k]=t2.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			elif w2==0:
				out_dict[k]=t1.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			if mode=="tensor1":
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[w1:w2,:]=t2[w1:w2,:]
				elif t1.dim()==3:
					t1[w1:w2,:,:]=t2[w1:w2,:,:]
				elif t1.dim()==4:
					t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
				elif t1.dim()==5:
					t1[w1:w2,:,:,:,:]=t2[w1:w2,:,:,:,:]
			else:
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[:,w1:w2]=t2[:,w1:w2]
				elif t1.dim()==3:
					t1[:,w1:w2,:]=t2[:,w1:w2,:]
				elif t1.dim()==4:
					t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
				elif t1.dim()==5:
					t1[:,w1:w2,:,:,:]=t2[:,w1:w2,:,:,:]
			out_dict[k]=t1.to(torch.bfloat16)
			del w1,w2

		save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
		data_dict.append(k)
		del w,out_dict,t1,t2
	del sd10,sd20
	flush()

	for k in keys1:
		key_count=key_count+1
		if win!=None:
			win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")

		out_dict={}
		try:
			t1=sd11.pop(k).to(torch.float32)
			t2=sd21.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

		k=box.teco(k)
		w=ws[29]

		if mode=="normal":
			out_dict[k]=((1-w)*t1+w*t2).to(torch.bfloat16)

		elif "tensor" in mode:
			w1=(1-w)/2
			w2=w
			w1=round(t1.size()[0]*w1)
			w2=round(t1.size()[0]*(w1+w2))
			if w1==0:
				out_dict[k]=t2.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			elif w2==0:
				out_dict[k]=t1.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			if mode=="tensor1":
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[w1:w2,:]=t2[w1:w2,:]
				elif t1.dim()==3:
					t1[w1:w2,:,:]=t2[w1:w2,:,:]
				elif t1.dim()==4:
					t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
				elif t1.dim()==5:
					t1[w1:w2,:,:,:,:]=t2[w1:w2,:,:,:,:]
			else:
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[:,w1:w2]=t2[:,w1:w2]
				elif t1.dim()==3:
					t1[:,w1:w2,:]=t2[:,w1:w2,:]
				elif t1.dim()==4:
					t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
				elif t1.dim()==5:
					t1[:,w1:w2,:,:,:]=t2[:,w1:w2,:,:,:]
			out_dict[k]=t1.to(torch.bfloat16)
			del w1,w2

		save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
		data_dict.append(k)
		del w,out_dict,t1,t2
	del sd11,sd21
	flush()

	for k in keys2:
		key_count=key_count+1
		if win!=None:
			win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")

		out_dict={}
		try:
			t1=sd12.pop(k).to(torch.float32)
			t2=sd22.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

		k=box.teen(k)
		w=ws[0]

		if mode=="normal":
			out_dict[k]=((1-w)*t1+w*t2).to(torch.bfloat16)

		elif "tensor" in mode:
			w1=(1-w)/2
			w2=w
			w1=round(t1.size()[0]*w1)
			w2=round(t1.size()[0]*(w1+w2))
			if w1==0:
				out_dict[k]=t2.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			elif w2==0:
				out_dict[k]=t1.to(torch.bfloat16)
				save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
				del w,out_dict,t1,t2,w1,w1
				continue
			if mode=="tensor1":
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[w1:w2,:]=t2[w1:w2,:]
				elif t1.dim()==3:
					t1[w1:w2,:,:]=t2[w1:w2,:,:]
				elif t1.dim()==4:
					t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
				elif t1.dim()==5:
					t1[w1:w2,:,:,:,:]=t2[w1:w2,:,:,:,:]
			else:
				if t1.dim()==1:
					t1[w1:w2]=t2[w1:w2]
				elif t1.dim()==2:
					t1[:,w1:w2]=t2[:,w1:w2]
				elif t1.dim()==3:
					t1[:,w1:w2,:]=t2[:,w1:w2,:]
				elif t1.dim()==4:
					t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
				elif t1.dim()==5:
					t1[:,w1:w2,:,:,:]=t2[:,w1:w2,:,:,:]
			out_dict[k]=t1.to(torch.bfloat16)
			del w1,w2

		save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
		data_dict.append(k)
		del w,out_dict,t1,t2
	del sd12,sd22
	flush()

	for k in keys3:
		key_count=key_count+1
		if win!=None:
			win["info"].update("merging "+str(key_count)+"/"+str(dict_sum))
		else:
			print("\rmerging "+str(key_count)+"/"+str(dict_sum),end="")

		out_dict={}
		try:
			t1=sd13.pop(k).to(torch.float32)
			t2=sd23.pop(k).to(torch.float32)
		except:
			if win==None:
				print("Unsupported Anima checkpoint key: "+k)
			else:
				win["RUN"].Update(disabled=False)
				win["info"].update("Unsupported Anima checkpoint key: "+k)
			shutil.rmtree(os.getcwd()+"/safe_temp")
			return

		k=box.vae(k)
		w=ws[0]

		if v==1:
			out_dict[k]=t1.to(torch.bfloat16)
		elif v==2:
			out_dict[k]=t2.to(torch.bfloat16)
		else:
			if mode=="normal":
				out_dict[k]=((1-w)*t1+w*t2).to(torch.bfloat16)

			elif "tensor" in mode:
				w1=(1-w)/2
				w2=w
				w1=round(t1.size()[0]*w1)
				w2=round(t1.size()[0]*(w1+w2))
				if w1==0:
					out_dict[k]=t2.to(torch.bfloat16)
					save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
					del w,out_dict,t1,t2,w1,w1
					continue
				elif w2==0:
					out_dict[k]=t1.to(torch.bfloat16)
					save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
					del w,out_dict,t1,t2,w1,w1
					continue
				if mode=="tensor1":
					if t1.dim()==1:
						t1[w1:w2]=t2[w1:w2]
					elif t1.dim()==2:
						t1[w1:w2,:]=t2[w1:w2,:]
					elif t1.dim()==3:
						t1[w1:w2,:,:]=t2[w1:w2,:,:]
					elif t1.dim()==4:
						t1[w1:w2,:,:,:]=t2[w1:w2,:,:,:]
					elif t1.dim()==5:
						t1[w1:w2,:,:,:,:]=t2[w1:w2,:,:,:,:]
				else:
					if t1.dim()==1:
						t1[w1:w2]=t2[w1:w2]
					elif t1.dim()==2:
						t1[:,w1:w2]=t2[:,w1:w2]
					elif t1.dim()==3:
						t1[:,w1:w2,:]=t2[:,w1:w2,:]
					elif t1.dim()==4:
						t1[:,w1:w2,:,:]=t2[:,w1:w2,:,:]
					elif t1.dim()==5:
						t1[:,w1:w2,:,:,:]=t2[:,w1:w2,:,:,:]
				out_dict[k]=t1.to(torch.bfloat16)
				del w1,w2
		save_file(out_dict,os.getcwd()+"/safe_temp/"+k+".safetensors")
		data_dict.append(k)
		del w,out_dict,t1,t2
	del sd13,sd23
	flush()

	if win==None:
		print("")

	if win==None:
		print("making output")
	else:
		win["info"].update("making output")

	save_safe(data_dict,out_path,os.getcwd()+"/safe_temp")
		
	f=open(out_path.replace(".safetensors",".txt"),"w")
	for i in range(len(ckpts)):
		f.write("ckpt"+str(i+1)+" : "+ckpts[i]+"\n")
	f.write("weight : "+str(ws)+"\n")
	f.close()

	if win==None:
		print(out_path)
	else:
		win["RUN"].Update(disabled=False)
		win["info"].update("fin")

def gui():
	import FreeSimpleGUI as sg
	import tkinter as tk
	import pyperclip
	import threading

	sg.theme('TealMono')
	
	keys=["ckpt1","ckpt2","out"]
	for i in range(31):
		keys.append("w"+str(i))

	grp_rclick_menu={}
	for key in keys:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]

	lay=[
		[
			sg.Text("ckpt1"), sg.Input(key="ckpt1",right_click_menu=grp_rclick_menu["ckpt1"]),sg.Radio("vae",key="v1",group_id='destination'),
		],
		[
			sg.FileBrowse("FileBrowse",key="file_browse1", enable_events=True, file_types=(('ckpt file', '.safetensors'),)),
			sg.FolderBrowse("FolderBrowse",key="folder_browse1", enable_events=True)
		],
		[
			sg.Text("ckpt2"), sg.Input(key="ckpt2",right_click_menu=grp_rclick_menu["ckpt2"]),sg.Radio("vae",key="v2",group_id='destination'),
		],
		[
			sg.FileBrowse("FileBrowse",key="file_browse2", enable_events=True, file_types=(('ckpt file', '.safetensors'),)),
			sg.FolderBrowse("FolderBrowse",key="folder_browse2", enable_events=True)
		],
		[sg.Checkbox('BLOCK', key='block',default=False,enable_events=True)],
		[sg.Text("weight of ckpt2",key="w_text")],
		[sg.Input(key="w0",right_click_menu=grp_rclick_menu["w0"])],
		[
			sg.Frame("BASE",[[sg.Input(key="w1",right_click_menu=grp_rclick_menu["w1"], size=(10, 1))]],key="base"),
			sg.Frame("BLOCK00",[[sg.Input(key="w2",right_click_menu=grp_rclick_menu["w2"], size=(10, 1))]],key="b0"),
			sg.Frame("BLOCK01",[[sg.Input(key="w3",right_click_menu=grp_rclick_menu["w3"], size=(10, 1))]],key="b1"),
			sg.Frame("BLOCK02",[[sg.Input(key="w4",right_click_menu=grp_rclick_menu["w4"], size=(10, 1))]],key="b2"),
			sg.Frame("BLOCK03",[[sg.Input(key="w5",right_click_menu=grp_rclick_menu["w5"], size=(10, 1))]],key="b3"),
			sg.Frame("BLOCK04",[[sg.Input(key="w6",right_click_menu=grp_rclick_menu["w6"], size=(10, 1))]],key="b4"),
			sg.Frame("BLOCK05",[[sg.Input(key="w7",right_click_menu=grp_rclick_menu["w7"], size=(10, 1))]],key="b5"),
			sg.Frame("BLOCK06",[[sg.Input(key="w8",right_click_menu=grp_rclick_menu["w8"], size=(10, 1))]],key="b6"),
			sg.Frame("BLOCK07",[[sg.Input(key="w9",right_click_menu=grp_rclick_menu["w9"], size=(10, 1))]],key="b7"),
			sg.Frame("BLOCK08",[[sg.Input(key="w10",right_click_menu=grp_rclick_menu["w10"], size=(10, 1))]],key="b8"),
		],
		[
			sg.Frame("BLOCK09",[[sg.Input(key="w11",right_click_menu=grp_rclick_menu["w11"], size=(10, 1))]],key="b9"),
			sg.Frame("BLOCK10",[[sg.Input(key="w12",right_click_menu=grp_rclick_menu["w12"], size=(10, 1))]],key="b10"),
			sg.Frame("BLOCK11",[[sg.Input(key="w13",right_click_menu=grp_rclick_menu["w13"], size=(10, 1))]],key="b11"),
			sg.Frame("BLOCK12",[[sg.Input(key="w14",right_click_menu=grp_rclick_menu["w14"], size=(10, 1))]],key="b12"),
			sg.Frame("BLOCK13",[[sg.Input(key="w15",right_click_menu=grp_rclick_menu["w15"], size=(10, 1))]],key="b13"),
			sg.Frame("BLOCK14",[[sg.Input(key="w16",right_click_menu=grp_rclick_menu["w16"], size=(10, 1))]],key="b14"),
			sg.Frame("BLOCK15",[[sg.Input(key="w17",right_click_menu=grp_rclick_menu["w17"], size=(10, 1))]],key="b15"),
			sg.Frame("BLOCK16",[[sg.Input(key="w18",right_click_menu=grp_rclick_menu["w18"], size=(10, 1))]],key="b16"),
			sg.Frame("BLOCK17",[[sg.Input(key="w19",right_click_menu=grp_rclick_menu["w19"], size=(10, 1))]],key="b17"),
			sg.Frame("BLOCK18",[[sg.Input(key="w20",right_click_menu=grp_rclick_menu["w20"], size=(10, 1))]],key="b18"),
		],
		[
			sg.Frame("BLOCK19",[[sg.Input(key="w21",right_click_menu=grp_rclick_menu["w21"], size=(10, 1))]],key="b19"),
			sg.Frame("BLOCK20",[[sg.Input(key="w22",right_click_menu=grp_rclick_menu["w22"], size=(10, 1))]],key="b20"),
			sg.Frame("BLOCK21",[[sg.Input(key="w23",right_click_menu=grp_rclick_menu["w23"], size=(10, 1))]],key="b21"),
			sg.Frame("BLOCK22",[[sg.Input(key="w24",right_click_menu=grp_rclick_menu["w24"], size=(10, 1))]],key="b22"),
			sg.Frame("BLOCK23",[[sg.Input(key="w25",right_click_menu=grp_rclick_menu["w25"], size=(10, 1))]],key="b23"),
			sg.Frame("BLOCK24",[[sg.Input(key="w26",right_click_menu=grp_rclick_menu["w26"], size=(10, 1))]],key="b24"),
			sg.Frame("BLOCK25",[[sg.Input(key="w27",right_click_menu=grp_rclick_menu["w27"], size=(10, 1))]],key="b25"),
			sg.Frame("BLOCK26",[[sg.Input(key="w28",right_click_menu=grp_rclick_menu["w28"], size=(10, 1))]],key="b26"),
			sg.Frame("BLOCK27",[[sg.Input(key="w29",right_click_menu=grp_rclick_menu["w29"], size=(10, 1))]],key="b27"),
			sg.Frame("LLM",[[sg.Input(key="w30",right_click_menu=grp_rclick_menu["w30"], size=(10, 1))]],key="llm"),
		],
		[
			sg.Radio('NORMAL', key='normal',default=True,group_id='destination'),
			sg.Radio('TENSOR1', key='tensor1',default=False,group_id='destination'),
			sg.Radio('TENSOR2', key='tensor2',default=False,group_id='destination')
		],
		[sg.Checkbox('full file', key='ff')],
		[sg.Text("output path"), sg.Input(key="out",right_click_menu=grp_rclick_menu["out"]),sg.FileSaveAs(file_types=(('ckpt file', '.safetensors'),))],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Merge Ckpt Anima', lay)

	def lay_che(b,win):
		win["ckpt1"].hide_row()
		win["file_browse1"].hide_row()
		win["ckpt2"].hide_row()
		win["file_browse2"].hide_row()
		win["block"].hide_row()
		win["w_text"].hide_row()
		win["w0"].hide_row()
		win["base"].hide_row()
		win["b9"].hide_row()
		win["b19"].hide_row()
		win["normal"].hide_row()
		win["ff"].hide_row()
		win["out"].hide_row()
		win["info"].hide_row()
		win["RUN"].hide_row()
		win["ckpt1"].unhide_row()
		win["file_browse1"].unhide_row()
		win["ckpt2"].unhide_row()
		win["file_browse2"].unhide_row()
		win["block"].unhide_row()
		win["w_text"].unhide_row()
		if b:
			win["base"].unhide_row()
			win["b9"].unhide_row()
			win["b19"].unhide_row()
		else:
			win["w0"].unhide_row()
		win["normal"].unhide_row()
		win["ff"].unhide_row()
		win["out"].unhide_row()
		win["info"].unhide_row()
		win["RUN"].unhide_row()

	event, values = window.read(timeout=0)
	lay_che(False,window)

	while True:
		event, values = window.read()
			
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["out"]!="" and values["ckpt1"]!="" and values["ckpt2"]!="":
				ckpts=[values["ckpt1"],values["ckpt2"]]
				out_path=values["out"]
				if values["block"]:
					weights=[]
					weight=0.5
					for i in range(30):
						try:
							weights.append(float(values["w"+str(i+1)]))
						except:
							weights.append(weight)
						weight=weights[-1]
						window["w"+str(i+1)].update(str(weight))
				else:
					try:
						weight=float(values["w0"])
					except:
						weight=0.5
					window["w0"].update(str(weight))
					weights=[]
					for i in range(30):
						weights.append(weight)

				if values["normal"]:
					mode="normal"
				elif values["tensor1"]:
					mode="tensor1"
				else:
					mode="tensor2"

				ff=values["ff"]

				if values["v1"]:
					vae=1
				elif values["v2"]:
					vae=2
				else:
					vae=0

				thread1 = threading.Thread(target=mergeckpt,args=(ckpts,weights,out_path,mode,ff,window,vae))
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
		elif event=="block":
			lay_che(values["block"],window)
		elif event=="file_browse1":
			window["ckpt1"].update(values["file_browse1"])
		elif event=="folder_browse1":
			window["ckpt1"].update(values["folder_browse1"])
		elif event=="file_browse2":
			window["ckpt2"].update(values["file_browse2"])
		elif event=="folder_browse2":
			window["ckpt2"].update(values["folder_browse2"])

	window.close()
