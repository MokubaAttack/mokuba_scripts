import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"

path=__file__.replace("\\","/")
path2=path.replace("mokuba_scripts/__init__.py","basicsr/data/degradations.py")
f=open(path2,"r")
data=[]
for line in f:
	if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in line:
		line=line.replace(
			"from torchvision.transforms.functional_tensor import rgb_to_grayscale",
			"from torchvision.transforms.functional import rgb_to_grayscale"
			)
	data+=[line]
f.close()
f=open(path2,"w")
for line in data:
	f.write(line)
f.close()

import requests
url2="https://raw.githubusercontent.com/huggingface/diffusers/e11810aa9123a4d924021fe6604838a2d5305a94/src/diffusers/modular_pipelines/anima/modular_blocks_anima.py"
path2=path.replace("mokuba_scripts/__init__.py","diffusers/modular_pipelines/anima/modular_blocks_anima.py")
response = requests.get(url2)
with open(path2, 'wb') as f:
	f.write(response.content)
url2="https://raw.githubusercontent.com/huggingface/diffusers/e11810aa9123a4d924021fe6604838a2d5305a94/src/diffusers/modular_pipelines/anima/before_denoise.py"
path2=path.replace("mokuba_scripts/__init__.py","diffusers/modular_pipelines/anima/before_denoise.py")
response = requests.get(url2)
with open(path2, 'wb') as f:
	f.write(response.content)
url2="https://raw.githubusercontent.com/huggingface/diffusers/e11810aa9123a4d924021fe6604838a2d5305a94/src/diffusers/modular_pipelines/anima/encoders.py"
path2=path.replace("mokuba_scripts/__init__.py","diffusers/modular_pipelines/anima/encoders.py")
response = requests.get(url2)
with open(path2, 'wb') as f:
	f.write(response.content)

try:
	from .common.dl import (
		dlc,
		dlk
	)
	from .notebook.workflow import (
		mokusdxl,
		mokuani,
		mokusd
	)
	from .common.flush import reset_func
except:
	pass

try:
	from .gui.animagui import animagui
	from .gui.sdxlgui import sdxlgui
except:
	pass

from .tool import (
	merge_ckpt_anima,
	merge_ckpt_sdxl,
	make_safetensors_anima,
	make_safetensors_sdxl,
	merge_lora_anima,
	merge_lora_sdxl,
	subtract_ckpt_anima,
	subtract_ckpt_sdxl
)

data="""\
import os
os.environ["HF_HOME"]=os.getcwd()+"/pipecache"

try:
	from .common.dl import (
		dlc,
		dlk
	)
	from .notebook.workflow import (
		mokusdxl,
		mokuani,
		mokusd
	)
	from .common.flush import reset_func
except:
	pass

try:
	from .gui.animagui import animagui
	from .gui.sdxlgui import sdxlgui
except:
	pass

from .tool import (
	merge_ckpt_anima,
	merge_ckpt_sdxl,
	make_safetensors_anima,
	make_safetensors_sdxl,
	merge_lora_anima,
	merge_lora_sdxl,
	subtract_ckpt_anima,
	subtract_ckpt_sdxl
)
"""
f=open(__file__,"w")
f.write(data)
f.close()
