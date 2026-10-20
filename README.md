# mokuba_scripts
These are scripts that I use when I create images in diffusers and make models.  
These are scripts using two gpus in notebooks.
## How to install
```
git clone -b multi https://github.com/MokubaAttack/mokuba_scripts.git
cd mokuba_scripts
pip install .[cuda]
```
## Scripts
- ### mokuani  
  It is a scripts that make images by anima model.
  ```
  mokuani2(
		loras = [],
		#list of lora file path or version id of civitai
		#When you input version id of civitai, token is needed.
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
		#checkpoint file path or version id of civitai
		#When you input version id of civitai, token is needed.
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
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
		token = "",
		#civitai token
  )
  ```
- ### mokusdxl  
  It is a scripts that make images by sdxl1.0 model.
  ```
  mokusdxl2(
		loras = [],
		#list of lora file path or version id of civitai
		#When you input version id of civitai, token is needed.
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
		#checkpoint file path or version id of civitai
		#When you input version id of civitai, token is needed.
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
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
		pos_emb = [],
		#list of positive embedding file or version id of civitai
		#When you input version id of civitai, token is needed.
		neg_emb = [],
		#list of negative embedding file or version id of civitai
		#When you input version id of civitai, token is needed.
		vae_safe = "",
		#vae file path or version id of civitai
		#When you input version id of civitai, token is needed.
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
		token = "",
		#civitai token
  )
  ```
- ### mokusd  
  It is a scripts that make images by sd1.5 model.
  ```
  mokusd2(
		loras = [],
		#list of lora file path or version id of civitai
		#When you input version id of civitai, token is needed.
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
		#checkpoint file path or version id of civitai
		#When you input version id of civitai, token is needed.
		url = [],
		#dropbox infomation [ App key, App secret, Refresh_token]
		#If you input it, images are sent to dropbox.
		dtype = "f32",
		#calculation accuracy
		#f32, f16, bf16
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
		pos_emb = [],
		#list of positive embedding file or version id of civitai
		#When you input version id of civitai, token is needed.
		neg_emb = [],
		#list of negative embedding file or version id of civitai
		#When you input version id of civitai, token is needed.
		vae_safe = "",
		#vae file path or version id of civitai
		#When you input version id of civitai, token is needed.
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
		token = "",
		#civitai token
  )
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
