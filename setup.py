from setuptools import (
	setup,
	find_packages
)
import sys
import os

if not(os.path.exists("temp_mokuba_scripts")):
	os.mkdir("temp_mokuba_scripts")

try:
	import torch
except:
	sys.path=sys.path+["temp_mokuba_scripts/Lib/site-packages"]

try:
	import torch
except:
	for p in sys.path:
		p=p.replace("\\","/")
		if "/pip/_vendor/pyproject_hooks/_in_process" in p:
			p=p.removesuffix("/pip/_vendor/pyproject_hooks/_in_process")
			sys.path=[p]+sys.path

	from pip._internal.cli.main import main as _main
	_main(["install","torch==2.11.0",'--prefix=temp_mokuba_scripts'])
	sys.path=sys.path+["temp_mokuba_scripts/Lib/site-packages"]
	import torch

v=str(sys.version_info[1])
p=sys.platform
if p.startswith("linux"):
	p="manylinux_2_28_x86_64"
	tri="triton==3.6.0"
else:
	p="win_amd64"
	tri="triton-windows==3.6.0.post26"

if torch.cuda.is_available():
	nb_need=[
		"ipython",
		"torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchao==0.13.0",
		"torchaudio @ https://download-r2.pytorch.org/whl/cu128/torchaudio-2.11.0%2Bcu128-cp3"+v+"-cp3"+v+"-"+p+".whl",
		tri,
	]
elif torch.xpu.is_available():
	nb_need=[
		"ipython",
		"torch @ https://download-r2.pytorch.org/whl/xpu/torch-2.11.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchvision @ https://download-r2.pytorch.org/whl/xpu/torchvision-0.26.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchao==0.13.0",
		"torchaudio @ https://download-r2.pytorch.org/whl/xpu/torchaudio-2.11.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"triton-xpu @ https://download-r2.pytorch.org/whl/triton_xpu-3.6.0-cp3"+v+"-cp3"+v+"-"+p+".whl",
	]
else:
	nb_need=[
		"ipython",
		"torch==2.11.0",
		"torchvision==0.26.0",
		"torchao==0.13.0",
		"torchaudio==2.11.0",
		tri,
	]

if torch.xpu.is_available():
	gui_need=[
		"FreeSimpleGUI",
		"pyperclip",
		"torch @ https://download-r2.pytorch.org/whl/xpu/torch-2.11.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchvision @ https://download-r2.pytorch.org/whl/xpu/torchvision-0.26.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"triton-xpu @ https://download-r2.pytorch.org/whl/triton_xpu-3.6.0-cp3"+v+"-cp3"+v+"-"+p+".whl",
	]
elif torch.cuda.is_available():
	gui_need=[
		"FreeSimpleGUI",
		"pyperclip",
		"torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp3"+v+"-cp3"+v+"-"+p+".whl",
		"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp3"+v+"-cp3"+v+"-"+p+".whl",
		tri,
	]
else:
	gui_need=[
		"FreeSimpleGUI",
		"pyperclip",
		"torch==2.11.0",
		"torchvision==0.26.0",
		tri,
	]

setup(
	name='mokuba_scripts',
	version='1.0.0',
	packages=find_packages(),
	include_package_data=True,
	description='This is a script that I use when I create images by diffusers.',
	long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
	license='BSD-3-Clause',
	classifiers=[
		'License :: OSI Approved :: BSD License',
		'Programming Language :: Python :: 3.12',
	],
	install_requires=[
		"compel>=2.4.0",
		"diffusers==0.40.0",
		"basicsr @ https://github.com/MokubaAttack/mokuba_scripts/raw/refs/heads/main/basicsr_copy/basicsr-1.4.2.tar.gz",
		"realesrgan",
		"lycoris-lora==4.0.0",
		"piexif",
		"transformers==5.14.1",
		"optimum-quanto",
		"accelerate",
		"PEFT",
		"dropbox",
	],
	extras_require={
		"nb":nb_need,
		"gui":gui_need,
	},
)
