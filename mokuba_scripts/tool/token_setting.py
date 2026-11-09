import json
import pyperclip
import FreeSimpleGUI as sg
import os
import tkinter as tk

dv={
	"civitai_token":"",
	"kaggle_token":"",
	"kaggle_username":"",
	"huggingface_token":"",
}
def save_token(vs):
	out_sd={}
	for key in dv:
		if vs[key]!="":
			out_sd[key]=vs[key]
	f=open("pipecache/token.json","w")
	json.dump(out_sd,f,indent=2)
	f.close()
	
def load_token():
	if os.path.exists("pipecache/token.json"):
		f=open("pipecache/token.json","r")
		sd=json.load(f)
		f.close()
	else:
		sd={}
	for key in dv:
		if not(key in sd):
			sd[key]=dv[key]
	return sd

def gui():
	sd=load_token()

	grp_rclick_menu={}
	for key in sd:
		grp_rclick_menu[key]=[
			"",
			[
				"-copy-::"+key,"-cut-::"+key,"-paste-::"+key
			]
		]

	layout=[
		[sg.Text("civitai_token"), sg.Input(default_text=sd["civitai_token"],key="civitai_token",right_click_menu=grp_rclick_menu["civitai_token"])],
		[sg.Text("kaggle_username"), sg.Input(default_text=sd["kaggle_username"],key="kaggle_username",right_click_menu=grp_rclick_menu["kaggle_username"])],
		[sg.Text("kaggle_token"), sg.Input(default_text=sd["kaggle_token"],key="kaggle_token",right_click_menu=grp_rclick_menu["kaggle_token"])],
		[sg.Text("huggingface_token"), sg.Input(default_text=sd["huggingface_token"],key="huggingface_token",right_click_menu=grp_rclick_menu["huggingface_token"])],
		[sg.Button('SAVE', key='SAVE'),sg.Button('EXIT', key='EXIT')]
	]

	window = sg.Window('Token Setting', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			break
		elif event=="EXIT":
			break
		elif event=="SAVE":
			save_token(values)
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