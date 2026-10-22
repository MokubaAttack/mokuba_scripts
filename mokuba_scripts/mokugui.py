import FreeSimpleGUI as sg

tools=[
	"mokuanima",
	"mokusd",
	"mokusdxl",
	"merge_ckpt_anima",
	"merge_ckpt_sdxl",
	"make_safetensors_anima",
	"make_safetensors_sdxl",
	"merge_lora_anima",
	"merge_lora_sdxl",
	"subtract_ckpt_anima",
	"subtract_ckpt_sdxl",
	"change_dim",
	"get_vae",
	"accuracy",
	"plus_metadata",
	"civitai_dl",
	"kaggle_dl",
	"imgup",
]

def main():
	layout=[
		[sg.Text("Tool Name")],
		[sg.Combo(default_value="mokuanima",values=tools,key="tool")],
		[sg.Button('RUN', key='RUN'),sg.Button('EXIT', key='EXIT')]
	]
	window = sg.Window('mokugui', layout)

	while True:
		event, values = window.read()
		if event == sg.WINDOW_CLOSED:
			name=None
			break
		elif event=="EXIT":
			name=None
			break
		elif event=="RUN":
			name=values["tool"]
			break
	window.close()

	if name!=None:
		if name=="mokuanima":
			from .gui.animagui import animagui
			box=animagui()
			box.gui()
		elif name=="mokusd":
			from .gui.sdgui import sdgui
			box=sdgui()
			box.gui()
		elif name=="mokusdxl":
			from .gui.sdxlgui import sdxlgui
			box=sdxlgui()
			box.gui()
		elif name=="merge_ckpt_anima":
			from .tool import merge_ckpt_anima
			merge_ckpt_anima.gui()
		elif name=="merge_ckpt_sdxl":
			from .tool import merge_ckpt_sdxl
			merge_ckpt_sdxl.gui()
		elif name=="make_safetensors_anima":
			from .tool import make_safetensors_anima
			make_safetensors_anima.gui()
		elif name=="make_safetensors_sdxl":
			from .tool import make_safetensors_sdxl
			make_safetensors_sdxl.gui()
		elif name=="merge_lora_anima":
			from .tool import merge_lora_anima
			merge_lora_anima.gui()
		elif name=="merge_lora_sdxl":
			from .tool import merge_lora_sdxl
			merge_lora_sdxl.gui()
		elif name=="subtract_ckpt_anima":
			from .tool import subtract_ckpt_anima
			subtract_ckpt_anima.gui()
		elif name=="subtract_ckpt_sdxl":
			from .tool import subtract_ckpt_sdxl
			subtract_ckpt_sdxl.gui()
		elif name=="change_dim":
			from .tool import change_dim
			change_dim.gui()
		elif name=="get_vae":
			from .tool import get_vae
			get_vae.gui()
		elif name=="accuracy":
			from .tool import accuracy
			accuracy.gui()
		elif name=="plus_metadata":
			from .tool import plus_metadata
			plus_metadata.gui()
		elif name=="civitai_dl":
			from .tool import civitai_dl
			civitai_dl.gui()
		elif name=="kaggle_dl":
			from .tool import kaggle_dl
			kaggle_dl.gui()
		else:
			from .tool import imgup
			imgup.gui()
