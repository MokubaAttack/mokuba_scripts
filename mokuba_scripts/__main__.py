from .gui.animagui import animagui
from .gui.sdgui import sdgui
from .gui.sdxlgui import sdxlgui
from .tool import (
	merge_ckpt_anima,
	merge_ckpt_sdxl,
	make_safetensors_anima,
	make_safetensors_sdxl,
	merge_lora_anima,
	merge_lora_sdxl,
	subtract_ckpt_anima,
	subtract_ckpt_sdxl,
	change_dim,
	get_vae,
	accuracy,
	plus_metadata,
	civitai_dl,
	kaggle_dl,
	imgup
)
import sys

tools=[
	"animagui",
	"sdgui",
	"sdxlgui",
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

if __name__=="__main__":
	args = sys.argv
	if len(args)>1:
		if not(args[1] in tools):
			print("You must select one from the following tools.")
			for key in tools:
				print(key)
		else:
			if args[1]=="animagui":
				animagui.gui()
			elif args[1]=="sdgui":
				sdgui.gui()
			elif args[1]=="sdgui":
				sdxlgui.gui()
			elif args[1]=="merge_ckpt_anima":
				merge_ckpt_anima.gui()
			elif args[1]=="merge_ckpt_sdxl":
				merge_ckpt_sdxl.gui()
			elif args[1]=="make_safetensors_anima":
				make_safetensors_anima.gui()
			elif args[1]=="make_safetensors_sdxl":
				make_safetensors_sdxl.gui()
			elif args[1]=="merge_lora_anima":
				merge_lora_anima.gui()
			elif args[1]=="merge_lora_sdxl":
				merge_lora_sdxl.gui()
			elif args[1]=="subtract_ckpt_anima":
				subtract_ckpt_anima.gui()
			elif args[1]=="subtract_ckpt_sdxl":
				subtract_ckpt_sdxl.gui()
			elif args[1]=="change_dim":
				change_dim.gui()
			elif args[1]=="get_vae":
				get_vae.gui()
			elif args[1]=="accuracy":
				accuracy.gui()
			elif args[1]=="plus_metadata":
				plus_metadata.gui()
			elif args[1]=="civitai_dl":
				civitai_dl.gui()
			elif args[1]=="kaggle_dl":
				kaggle_dl.gui()
			else:
				imgup.gui()
	else:
		print("You must select one from the following tools.")
		for key in tools:
			print(key)
