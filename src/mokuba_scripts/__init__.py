import os
import tempfile
import json
import FreeSimpleGUI as sg

home_dir=os.environ.get('LOCALAPPDATA',os.path.expanduser("~"))+"/mokuba_config"
cache_dir=tempfile.gettempdir()+"/mokuba_cache"

os.environ["HF_HOME"]=cache_dir
if not(os.path.exists(cache_dir)):
	os.mkdir(cache_dir)
	
if not(os.path.exists(home_dir)):
	os.mkdir(home_dir)
if os.path.exists(home_dir+"/token.json"):
	f=open(home_dir+"/token.json","r")
	sd=json.load(f)
	f.close()
	if "huggingface_token" in sd:
		os.environ["HF_TOKEN"]=sd["huggingface_token"]
	
sg.theme('TealMono')

from .common.dl import (
	dlc,
	dlk
)
from .common.discord import (
	up_drop,
	down_drop
)
from .notebook.run import (
	mokuani2,
	mokusd2,
	mokusdxl2
)
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
	merge_lora_anima2,
	merge_lora_sdxl2,
	subtract_ckpt_anima,
	subtract_ckpt_sdxl,
	change_dim,
	get_vae,
	accuracy,
	plus_metadata,
	civitai_dl,
	kaggle_dl,
	imgup,
	token_setting
)
