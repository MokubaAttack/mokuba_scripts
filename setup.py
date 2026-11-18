from setuptools import (
	setup,
	find_packages
)
import sys

def extras_need(w):
	v=str(sys.version_info[1])
	p=sys.platform
	if p.startswith("linux"):
		p="manylinux_2_28_x86_64"
		tri="triton @ https://download-r2.pytorch.org/whl/triton-3.8.0-cp3"+v+"-cp3"+v+"-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
	else:
		p="win_amd64"
		tri="triton-windows==3.8.0.post28"

	if w=="cuda":
		need=[
			"torch @ https://download-r2.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp3"+v+"-cp3"+v+"-"+p+".whl",
			"torchvision @ https://download-r2.pytorch.org/whl/cu130/torchvision-0.29.0%2Bcu130-cp3"+v+"-cp3"+v+"-"+p+".whl",
			tri,
		]
	elif w=="xpu":
		need=[
			"torch @ https://download-r2.pytorch.org/whl/xpu/torch-2.14.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
			"torchvision @ https://download-r2.pytorch.org/whl/xpu/torchvision-0.29.0%2Bxpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
			"triton-xpu @ https://download-r2.pytorch.org/whl/triton_xpu-3.8.0-cp3"+v+"-cp3"+v+"-"+p+".whl",
		]
	else:
		need=[
			"torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
			"torchvision @ https://download-r2.pytorch.org/whl/cpu/torchvision-0.29.0%2Bcpu-cp3"+v+"-cp3"+v+"-"+p+".whl",
			tri,
		]
	return need

setup(
	name='mokuba_scripts',
	version='1.2.1',
	author='mokuba_attack',
	author_email='mokuatta@nigge.rs',
	package_dir={"": "src"},
	packages=find_packages("src"),
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
		"diffusers @ git+https://github.com/MokubaAttack/diffusers.git",
		"basicsr @ git+https://github.com/MokubaAttack/BasicSR.git",
		"realesrgan",
		"lycoris-lora==4.0.0",
		"piexif",
		"transformers==5.22.1",
		"optimum-quanto",
		"accelerate",
		"PEFT",
		"dropbox",
		"ipython",
		"FreeSimpleGUI",
		"pyperclip",
		"requests==2.33.1",
	],
	extras_require={
		"cuda":extras_need(w="cuda"),
		"xpu":extras_need(w="xpu"),
		"cpu":extras_need(w="cpu"),
	},
	entry_points={
		'console_scripts': [
        	'mokugui = mokuba_scripts.mokugui:main',
        ]
	},
)
