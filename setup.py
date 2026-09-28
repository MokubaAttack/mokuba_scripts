from setuptools import setup, find_packages
import sys

def cuda_require(w):
	key="/pip/_vendor/pyproject_hooks/_in_process"
	paths=sys.path
	for p in paths:
		p=p.replace("\\","/")
		if key in p:
			p=p.removesuffix(key)
			sys.path=[p]+sys.path
			break
	try:
		import torch
	except:
		from pip._internal.cli.main import main as _main
		_main(["install","torch==2.11.0"])
		import torch
	v=str(sys.version_info[1])
	need=[]
	if w=="nb":
		if torch.cuda.is_available():
			need=[
				"ipython",
				"torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp3"+v+"-cp3"+v+"-manylinux_2_28_x86_64.whl",
				"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp3"+v+"-cp3"+v+"-manylinux_2_28_x86_64.whl",
				"torchao==0.13.0",
				"torchaudio @ https://download-r2.pytorch.org/whl/cu128/torchaudio-2.11.0%2Bcu128-cp3"+v+"-cp3"+v+"-manylinux_2_28_x86_64.whl",
			]
		else:
			need=[
				"ipython",
				"torch==2.11.0",
				"torchvision==0.26.0",
				"torchao==0.13.0",
				"torchaudio==2.11.0",
				"triton",
			]
	elif w=="gui":
		if torch.xpu.is_available():
			need=[
				"FreeSimpleGUI",
				"pyperclip",
				"torch @ https://download-r2.pytorch.org/whl/xpu/torch-2.11.0%2Bxpu-cp3"+v+"-cp3"+v+"-win_amd64.whl",
				"torchvision @ https://download-r2.pytorch.org/whl/xpu/torchvision-0.26.0%2Bxpu-cp3"+v+"-cp3"+v+"-win_amd64.whl",
				"triton-xpu @ https://download-r2.pytorch.org/whl/triton_xpu-3.7.0-cp3"+v+"-cp3"+v+"-win_amd64.whl",
			]
		else:
			need=[
				"FreeSimpleGUI",
				"pyperclip",
				"torch==2.11.0",
				"torchvision==0.26.0",
			]
	return need

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
		"nb":cuda_require("nb"),
		"gui":cuda_require("gui"),
	},
)
