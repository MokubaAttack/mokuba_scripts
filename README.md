# mokuba_scripts
These are scripts that I use when I create images in diffusers and make models.
## How to install
- colab
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
cd basicsr_copy
pip install .
cd ..
pip install .[colab]
```
- kaggle
```
git clone https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[kaggle]
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
#### merge_ckpt
It is a script that merge checkpoints. This script supports block merge and tensor merge.
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
#### make_safetensors
It is a script that burns a vae and loras in a checkpoint. This script is based on [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) of huggingface/diffusers.
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
#### merge_lora
It is a script that merge lora by SVD. This script is based on [svd_merge_lora.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py) of kohya-ss/sd-scripts.
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
  merge_lora_anima.gui()
  ```
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
#### subtract_ckpt
It is a script that make difference of checkpoints into lora. This script is based on [extract_lora_from_models.py](https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py) of kohya-ss/sd-scripts.
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
#### change_dim
It is a script that changes dim of lora by svd.
- change_dim
  ```
  change_dim.changedim(
		path="",
		#lora file path
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
  )
  ```
  ```
  change_dim.gui()
  ```
### notebook only
- #### mokuani  
  It is a scripts that make images by anima model.
  ```
  mokuani(
		loras = [],
		#list of lora file path
		lora_weights = [],
		#list of lora weight
		prompt = "",
		#prompt
		n_prompt = "",
		#negative prompt
		pic_number = 10,
		#number of output images
		gs = 7,
		#guidance_scale ( a parameter of StableDiffusion )
		step = 30,
		#num_inference_steps ( a parameter of StableDiffusion )
		sample = "",
		#scheduler type
		#FlowMatch_Euler, FlowMatch_LCM
		sgm = "",
		#noise schedule
		#karras, beta, exponential, normal
		seed = 0,
		#seed
		out_folder = "data",
		#output folder path
		base_safe = "base.safetensors",
		#checkpoint file path
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
		dev = "cuda",
		#calculation device
		#cuda, mps, xpu, cpu
		x = 1024,
		#width of output image
		y = 1024,
		#height of output image
		mode = 0,
		#working mode
		#0 : normal, 1 : hires.fix
		up = 1.5,
		#Hires upscale ( a parameter of hires.fix )
		Interpolation = "BILINEAR",
		#interpolation method of the upscaling
		#NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS
		#If you input pth file of ESRGAN, images are upscaled by ESRGAN.
		step2 = 15,
		#Hires steps ( a parameter of hires.fix )
		ss = 0.5,
		#denoising_strength ( a parameter of hires.fix )
		p = None,
		#If you input mokuanipipe object, you can use same pipeline without making the pipeline.
		ser = "colab",
		#In google colab, please input "colab". In kaggle, please input "kaggle".
		del_pipe = True,
		#If you choice True, mokuanipipe object is deleted and None is returned.
		si = True,
		#If you choice True, output images are shown in the output window.
  ) = mokuanipipe object
  ```
- #### mokusdxl  
  It is a scripts that make images by sdxl1.0 model.
  ```
  mokusdxl(
		loras = [],
		#list of lora file path
		lora_weights = [],
		#list of lora weight
		prompt = "",
		#prompt
		n_prompt = "",
		#negative prompt
		pic_number = 10,
		#number of output images
		gs = 7,
		#guidance_scale ( a parameter of StableDiffusion )
		step = 30,
		#num_inference_steps ( a parameter of StableDiffusion )
		sample = "",
		#scheduler type
		#Euler a, Euler, LMS, Heun, DPM2, DPM2 a, DPM++, DPM++ 2M, DPM++ SDE, DPM++ 2M SDE, DPM++ 3M SDE, DDIM, PLMS, UniPC, LCM
		sgm = "",
		#noise schedule
		#Karras, sgm_uniform, simple, exponential, beta
		seed = 0,
		#seed
		out_folder = "data",
		#output folder path
		base_safe = "base.safetensors",
		#checkpoint file path
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
		dev = "cuda",
		#calculation device
		#cuda, mps, xpu, cpu
		x = 1024,
		#width of output image
		y = 1024,
		#height of output image
		mode = 0,
		#working mode
		#0 : normal, 1 : hires.fix, 2 : hires.fix to tile upscaler
		up = 1.5,
		#Hires upscale ( a parameter of hires.fix )
		Interpolation = "BILINEAR",
		#interpolation method of the upscaling
		#NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS
		#If you input pth file of ESRGAN, images are upscaled by ESRGAN.
		step2 = 15,
		#Tile scale ( a parameter of Tile Upscaler)
		ss = 0.5,
		#denoising_strength ( a parameter of hires.fix and Tile Upscaler )
		p = None,
		#If you input mokusdxlpipe object, you can use same pipeline without making the pipeline.
		ser = "colab",
		#In google colab, please input "colab". In kaggle, please input "kaggle".
		del_pipe = True,
		#If you choice True, mokusdxlpipe object is deleted and None is returned.
		si = True,
		#If you choice True, output images are shown in the output window.
		pos_emb = [],
		#list of positive embedding files
		neg_emb = [],
		#list of negative embedding files
		vae_safe = "",
		#vae file path
		step3 = 20,
		#num_inference_steps for tile upscaler
		up2 = 1.5,
		#tile upscale ( a parameter of Tile Upscaler )
		ccs = 0,
		#controlnet_conditioning_scale ( a parameter of Tile Upscaler )
		#If ccs = 0, controlnet tile are not used.
		gpulowmem = False,
		#If you choose True, gpu memory isn't used much.
		freezeunet = False,
		#If you choose True, unet freeze to qfloat8.
		cs = 2,
		#clip_skip ( a parameter of StableDiffusion )
		qprompt = "masterpiece, best quality, ultra detailed",
		#prompt for tile upscaler
		qn_prompt = "worst quality, low quality, normal quality",
		#negative prompt for tile upscaler
  ) = mokusdxlpipe object
  ```
- #### mokusd  
  It is a scripts that make images by sd1.5 model.
  ```
  mokusd(
		loras = [],
		#list of lora file path
		lora_weights = [],
		#list of lora weight
		prompt = "",
		#prompt
		n_prompt = "",
		#negative prompt
		pic_number = 10,
		#number of output images
		gs = 7,
		#guidance_scale ( a parameter of StableDiffusion )
		step = 30,
		#num_inference_steps ( a parameter of StableDiffusion )
		sample = "",
		#scheduler type
		#Euler a, Euler, LMS, Heun, DPM2, DPM2 a, DPM++, DPM++ 2M, DPM++ SDE, DPM++ 2M SDE, DPM++ 3M SDE, DDIM, PLMS, UniPC, LCM
		sgm = "",
		#noise schedule
		#Karras, sgm_uniform, simple, exponential, beta
		seed = 0,
		#seed
		out_folder = "data",
		#output folder path
		base_safe = "base.safetensors",
		#checkpoint file path
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
		dev = "cuda",
		#calculation device
		#cuda, mps, xpu, cpu
		x = 1024,
		#width of output image
		y = 1024,
		#height of output image
		mode = 0,
		#working mode
		#0 : normal, 1 : hires.fix, 2 : hires.fix to tile upscaler
		up = 1.5,
		#Hires upscale ( a parameter of hires.fix )
		Interpolation = "BILINEAR",
		#interpolation method of the upscaling
		#NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS
		#If you input pth file of ESRGAN, images are upscaled by ESRGAN.
		step2 = 15,
		#Tile scale ( a parameter of Tile Upscaler)
		ss = 0.5,
		#denoising_strength ( a parameter of hires.fix and Tile Upscaler )
		p = None,
		#If you input mokusdpipe object, you can use same pipeline without making the pipeline.
		ser = "colab",
		#In google colab, please input "colab". In kaggle, please input "kaggle".
		del_pipe = True,
		#If you choice True, mokusdpipe object is deleted and None is returned.
		si = True,
		#If you choice True, output images are shown in the output window.
		pos_emb = [],
		#list of positive embedding files
		neg_emb = [],
		#list of negative embedding files
		vae_safe = "",
		#vae file path
		step3 = 20,
		#num_inference_steps for tile upscaler
		up2 = 1.5,
		#tile upscale ( a parameter of Tile Upscaler )
		ccs = 0,
		#controlnet_conditioning_scale ( a parameter of Tile Upscaler )
		#If ccs = 0, controlnet tile are not used.
		gpulowmem = False,
		#If you choose True, gpu memory isn't used much.
		freezeunet = False,
		#If you choose True, unet freeze to qfloat8.
		cs = 2,
		#clip_skip ( a parameter of StableDiffusion )
		qprompt = "masterpiece, best quality, ultra detailed",
		#prompt for tile upscaler
		qn_prompt = "worst quality, low quality, normal quality",
		#negative prompt for tile upscaler
  ) = mokusdpipe object
  ```
### gui only
- #### animagui  
  It is gui version of mokuani.
  ```
  animagui.gui()
  ```
- #### sdxlgui  
  It is gui version of mokusdxl.
  ```
  sdxlgui.gui()
  ```
- #### sdgui  
  It is gui version of mokusd.
  ```
  sdgui.gui()
  ```
- #### get_vae  
  It is a script that extracts a vae safetensors from a checkpoint safetensors.
  ```
  get_vae.gui()
  ```
- #### accuracy  
  It is a script that change accuracy of safetensors file.
  ```
  accuracy.gui()
  ```
- #### plus_metadata  
  It is a script that write metadata to PNG file and JPG file. That metadata is recognized in CivitAi.
  ```
  plus_metadata.gui()
  ```
- #### civitai_dl  
  It is a script that downloads data from CivitAi.
  ```
  civitai_dl.gui()
  ```
- #### kaggle_dl  
  It is a script that downloads data from Kaggle Dataset.
  ```
  kaggle_dl.gui()
  ```
- #### imgup  
  It is a script that makes jpg file and png file larger.
  ```
  imgup.gui()
  ```
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
