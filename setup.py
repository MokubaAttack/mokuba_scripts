from setuptools import setup, find_packages

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
		"diffusers==0.39.0",
		"realesrgan",
		"lycoris-lora",
		"piexif",
		"transformers==5.11.0",
		"optimum-quanto",
		"accelerate",
		"PEFT",
		"dropbox",
	],
	extras_require={
		"kaggle":[
			"ipython",
			"torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
			"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
			"torchao>=0.16.0",
		],
		"colab":[
			"ipython",
			"torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl",
			"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl",
			"torchao>=0.16.0",
		],
		"nbcpu":[
			"ipython",
			"torch==2.11.0",
			"torchvision==0.26.0",
			"torchao>=0.16.0",
		],
		"guicpu":[
			"FreeSimpleGUI",
			"pyperclip",
			"torch==2.11.0",
			"torchvision==0.26.0",
		],
		"guixpu":[
			"FreeSimpleGUI",
			"pyperclip",
			"torch @ https://download-r2.pytorch.org/whl/xpu/torch-2.11.0%2Bxpu-cp312-cp312-win_amd64.whl",
			"torchvision @ https://download-r2.pytorch.org/whl/xpu/torchvision-0.26.0%2Bxpu-cp312-cp312-win_amd64.whl",
			"triton-xpu @ https://download-r2.pytorch.org/whl/triton_xpu-3.7.0-cp312-cp312-win_amd64.whl",
		],
	},
)
