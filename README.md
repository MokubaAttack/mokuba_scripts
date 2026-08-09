# mokuba_scripts
These are scripts that I use when I create images in diffusers and make models.
## How to install
- notebook by cuda
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[nbcuda]
```
- notebook by cpu
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[nbcpu]
```
- gui by cpu
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[guicpu]
```
- gui by xpu
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[guixpu]
```
## Scripts
### common
- merge_ckpt_anima
  ```
  merge_ckpt_anima.mergeckpt(
		ckpts,
		#checkpoint files list 
		ws,
		#weights list of ckpts[1]
		#[BASE,BLOCK00,BLOCK01,BLOCK02,BLOCK03,...,BLOCK26,BLOCK27,LLM]
		out_path,
		#output file path
		mode = "normal",
		#merge mode
		#normal, tensor1, tensor2
		ff = True,
		#When you choose True, this program merges text_encoder and vae too.
		v = 0,
		#When v = 1, this program adopts vae of ckpts[0]. When v = 2, it adopts vae of ckpts[1]. When v = 0, it merge vae.
  )
  ```
  ```
  merge_ckpt_anima.gui()
  ```
  gui is opened.
- merge_ckpt_sdxl  
  ```
  merge_ckpt_sdxl.mergeckpt(
		ckpts,
		#checkpoint files list 
		weights,
		#weights list of ckpts[1]
		#[BASE,IN00,IN01,...,IN08,MID,OUT00,OUT01,...,OUT08]
		v,
		#When v = 1, this program adopts vae of ckpts[0]. When v = 2, it adopts vae of ckpts[1]. When v = 0, it merge vae.
		out_path,
		#output file path
		mode = "normal",
		#merge mode
		#normal, tensor1, tensor2, dare, mokuba
		dp = 0,
		#Dropout probability ( dare mode only )
		seed = 0,
		#seed ( dare mode only )
  )
  ```
  ```
  merge_ckpt_sdxl.gui()
  ```
  gui is opened.
- make_safetensors_anima
  ```
  make_safetensors_anima.makesafe(
		base_path,
		#checkpoint file path
		loras,
		#list of lora file path
		ws,
		#list of lora weight
		out_path,
		#output file path
		ff,
		#When you choose True, the output file contains text_encoder and vae too.
  )
  ```
  ```
  make_safetensors_anima.gui()
  ```
  gui is opened.
- make_safetensors_sdxl  
  ```
  make_safetensors_sdxl.makesafe(
		base_path,
		#checkpoint file path
		loras,
		#list of lora file path
		ws,
		#list of lora weight
		out_path,
		#output file path
		vae,
		#vae file path
  )
  ```
  ```
  make_safetensors_sdxl.gui()
  ```
  gui is opened.
- merge_lora_anima
  ```
  merge_lora_anima.mergelora(
		loras=[],
		#list of lora file path
		weights=[],
		#list of lora weight
		precision="float",
		#calculation accuracy
		save_precision="fp16",
		#output accuracy
		new_rank=16,
		#Specify rank of output LoRA
		new_conv_rank=None,
		#Specify rank of output LoRA for Conv2d 3x3
		device=None,
		#calculation device
		save_to=None,
		#output file path
		meta_dict=None,
		#metadata dictionary
		dof=False,
		#When you choose True, input lora files are deleted.
  )
  ```
  ```
  merge_lora_anima.gui()
  ```
  gui is opened.
- merge_lora_sdxl
  ```
  merge_lora_sdxl.mergelora(
		loras=[],
		#list of lora file path
		weights=[],
		#list of lora weight
		precision="float",
		#calculation accuracy
		save_precision="fp16",
		#output accuracy
		new_rank=16,
		#rank of output LoRA
		new_conv_rank=None,
		#rank of output LoRA for Conv2d 3x3
		device=None,
		#calculation device
		save_to=None,
		#output file path
		meta_dict=None,
		#metadata dictionary
		dof=False,
		#When you choose True, input lora files are deleted.
  )
  ```
  ```
  merge_lora_sdxl.gui()
  ```
  gui is opened. 
- subtract_ckpt_anima
  ```
  subtract_ckpt_anima.subtractckpt(
		ckpts,
		#checkpoint files list
		dim,
		#rank of output LoRA
		trans,
		#When you choose True, transfomer parts are output
		teco,
		#When you choose True, text_conditioner parts are output
		teen,
		#When you choose True, text_encoder parts are output
		out_path,
		#output file path
  )
  ```
  ```
  subtract_ckpt_anima.gui()
  ```
  gui is opened.
- subtract_ckpt_sdxl
  ```
  subtract_ckpt_sdxl.subtractckpt(
		ckpts,
		#checkpoint files list
		dim,
		#rank of output LoRA
		trans,
		#When you choose True, unet parts are output
		teen1,
		#When you choose True, text_encoder parts are output
		teen2,
		#When you choose True, text_encoder_2 parts are output
		out_path,
		#output file path
  )
  ```
  ```
  subtract_ckpt_sdxl.gui()
  ```
  gui is opened.
### notebook only
- mokuani
  ```
  mokuani(
		loras = [],
		lora_weights = [],
		prompt = "",
		n_prompt = "",
		pic_number = 10,
		gs = 7,
		step = 30,
		sample = "",
		sgm = "",
		seed = 0,
		out_folder = "data",
		base_safe = "base.safetensors",
		url = "",
		dtype = "f32",
		dev = "cuda",
		x = 1024,
		y = 1024,
		mode = 0,
		up = 1.5,
		Interpolation = "BILINEAR",
		step2 = 15,
		ss = 0.5,
		p = None,
		ser = "colab",
		del_pipe = True,
		si = True,
  )
  ```
- mokusdxl
  ```
  mokusdxl(
		loras = [],
		lora_weights = [],
		prompt = "",
		n_prompt = "",
		pic_number = 10,
		gs = 7,
		step = 30,
		sample = "",
		sgm = "",
		seed = 0,
		out_folder = "data",
		base_safe = "base.safetensors",
		url = "",
		dtype = "f32",
		dev = "cuda",
		x = 1024,
		y = 1024,
		mode = 0,
		up = 1.5,
		Interpolation = "BILINEAR",
		step2 = 15,
		ss = 0.5,
		p = None,
		ser = "colab",
		del_pipe = True,
		si = True,
		pos_emb = [],
		neg_emb = [],
		vae_safe = "",
		step3 = 20,
		up2 = 1.5,
		ccs = 0,
		gpulowmem = False,
		freezeunet = False,
		cs = 2,
		qprompt = "masterpiece, best quality, ultra detailed",
		qn_prompt = "worst quality, low quality, normal quality",
  )
  ```
- mokusd
  ```
  mokusd(
		loras = [],
		lora_weights = [],
		prompt = "",
		n_prompt = "",
		pic_number = 10,
		gs = 7,
		step = 30,
		sample = "",
		sgm = "",
		seed = 0,
		out_folder = "data",
		base_safe = "base.safetensors",
		url = "",
		dtype = "f32",
		dev = "cuda",
		x = 1024,
		y = 1024,
		mode = 0,
		up = 1.5,
		Interpolation = "BILINEAR",
		step2 = 15,
		ss = 0.5,
		p = None,
		ser = "colab",
		del_pipe = True,
		si = True,
		pos_emb = [],
		neg_emb = [],
		vae_safe = "",
		step3 = 20,
		up2 = 1.5,
		ccs = 0,
		gpulowmem = False,
		freezeunet = False,
		cs = 2,
		qprompt = "masterpiece, best quality, ultra detailed",
		qn_prompt = "worst quality, low quality, normal quality",
  )
  ```
### gui only
- animagui
  ```
  animagui.gui()
  ```
  gui is opened.
- sdxlgui
  ```
  sdxlgui.gui()
  ```
  gui is opened.
- get_vae
  ```
  get_vae.gui()
  ```
  gui is opened.
- accuracy
  ```
  accuracy.gui()
  ```
  gui is opened.
- plus_metadata
  ```
  plus_metadata.gui()
  ```
  gui is opened.
- civitai_dl
  ```
  civitai_dl.gui()
  ```
  gui is opened.
- kaggle_dl
  ```
  kaggle_dl.gui()
  ```
  gui is opened.
- imgup
  ```
  imgup.gui()
  ```
  gui is opened.
## Credits
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main)
- [hako-mikan/sd-webui-supermerger](https://github.com/hako-mikan/sd-webui-supermerger)
- [martyn/safetensors-merge-supermario](https://github.com/martyn/safetensors-merge-supermario)
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)
- [gokayfem/Tile-Upscaler](https://github.com/gokayfem/Tile-Upscaler)
- [lllyasviel/control_v11f1e_sd15_tile](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile)
- [OzzyGT/SDXL_Controlnet_Tile_Realistic](https://huggingface.co/OzzyGT/SDXL_Controlnet_Tile_Realistic)
- [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)
