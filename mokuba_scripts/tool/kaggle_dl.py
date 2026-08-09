import FreeSimpleGUI as sg
import tkinter as tk
import threading
import pyperclip
import json
import os

from ..common.dl import dlk

def token_input():
	keys=['token',"Username"]
	grp_rclick_menu={}
	for k in keys:
		grp_rclick_menu[k]=[
			"",
			[
				"-copy-::"+k,"-cut-::"+k,"-paste-::"+k
			]
		]

	layout=[
		[sg.Text("Kaggle Username"), sg.Input(key="Username",right_click_menu=grp_rclick_menu["Username"])],
		[sg.Text("Kaggle Token"), sg.Input(key="token",right_click_menu=grp_rclick_menu["token"])],
		[sg.Button('SAVE', key='SAVE'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Input Civitai Token', layout)

	while True:
		event, values = window.read()
			
		if event == sg.WINDOW_CLOSED:
			values["token"]=""
			break
		elif event=="EXIT":
			values["token"]=""
			break
		elif event=="SAVE":
			break
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

	if values["token"]!="" and values["Username"]!="":
		if not(os.path.exists(os.getcwd()+"/token.json")):
			d={
				"kaggle_token":values["token"],
				"kaggle_username":values["Username"],
			}
		else:
			with open(os.getcwd()+"/token.json","r") as f:
				d = json.load(f)
			d["kaggle_token"]=values["token"]
			d["kaggle_username"]=values["Username"]
		with open(os.getcwd()+"/token.json", 'w') as f:
			json.dump(d, f, indent=2)
		return True
	else:
		return False

def check():
	if not(os.path.exists(os.getcwd()+"/token.json")):
		return token_input()

	with open(os.getcwd()+"/token.json","r") as f:
		d = json.load(f)

	if not("kaggle_token" in d) or not("kaggle_username" in d):
		return token_input()

	return True

def run(i,path,win):
	win["RUN"].Update(disabled=True)
	try:
		with open(os.getcwd()+"/token.json","r") as f:
			d = json.load(f)
		t=d["kaggle_token"]
		username=d["kaggle_username"]
	except:
		win["info"].update("error")
		win["RUN"].Update(disabled=False)
		return
	try:
		dlk(i,username,t,path)
		win["info"].update("fin")
		win["RUN"].Update(disabled=False)
	except:
		win["info"].update("error")
		win["RUN"].Update(disabled=False)
		return

def gui():
	if check()==False:
		return
	keys=['output','id']
	grp_rclick_menu={}
	for k in keys:
		grp_rclick_menu[k]=[
			"",
			[
				"-copy-::"+k,"-cut-::"+k,"-paste-::"+k
			]
		]

	layout=[
		[sg.Text("output"), sg.Input(key="output",right_click_menu=grp_rclick_menu["output"]),sg.FolderBrowse()],
		[sg.Text("dataset title"), sg.Input(key="id",right_click_menu=grp_rclick_menu["id"])],
		[sg.Text("infomation",key="info")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Kaggle Download', layout,keep_on_top=True)

	while True:
		event, values = window.read()
			
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="RUN":
			if values["output"]!="" and values["id"]!="":
				path=values["output"]
				i=values["id"]
				thread1 = threading.Thread(target=run,args=(i,path,window))
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

	window.close()