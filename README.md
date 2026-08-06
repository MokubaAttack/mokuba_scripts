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
  merge_ckpt_anima.mergeckpt()
  ```
  ```
  merge_ckpt_anima.gui()
  ```
  gui is opened.
- merge_ckpt_sdxl  
  ```
  merge_ckpt_sdxl.mergeckpt()
  ```
  ```
  merge_ckpt_sdxl.gui()
  ```
  gui is opened.
- make_safetensors_anima
  ```
  make_safetensors_anima.makesafe()
  ```
  ```
  make_safetensors_anima.gui()
  ```
  gui is opened.
- make_safetensors_sdxl  
  ```
  make_safetensors_sdxl.makesafe()
  ```
  ```
  make_safetensors_sdxl.gui()
  ```
  gui is opened.
- merge_lora_anima
  ```
  merge_lora_anima.mergelora()
  ```
  ```
  merge_lora_anima.gui()
  ```
  gui is opened.
- merge_lora_sdxl
  ```
  merge_lora_sdxl.mergelora()
  ```
  ```
  merge_lora_sdxl.gui()
  ```
  gui is opened. 
- subtract_ckpt_anima
  ```
  subtract_ckpt_anima.subtractckpt()
  ```
  ```
  subtract_ckpt_anima.gui()
  ```
  gui is opened.
- subtract_ckpt_sdxl
  ```
  subtract_ckpt_sdxl.subtractckpt()
  ```
  ```
  subtract_ckpt_sdxl.gui()
  ```
  gui is opened.
### notebook only
- mokuani
- mokusdxl
- mokusd
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
