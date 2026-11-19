from setuptools import (
	setup,
	find_packages
)
import sys
import subprocess
import time
import os

realesrgan_version_file = 'src/submodules/Real-ESRGAN/realesrgan/version.py'
realesrgan_version='src/submodules/Real-ESRGAN/VERSION'
basicsr_version_file = 'src/submodules/BasicSR/basicsr/version.py'
basicsr_version='src/submodules/BasicSR/VERSION'
gfpgan_version_file = 'src/submodules/GFPGAN/gfpgan/version.py'
gfpgan_version = 'src/submodules/GFPGAN/VERSION'

def get_git_hash():

	def _minimal_ext_cmd(cmd):
		env = {}
		for k in ['SYSTEMROOT', 'PATH', 'HOME']:
			v = os.environ.get(k)
			if v is not None:
				env[k] = v
		env['LANGUAGE'] = 'C'
		env['LANG'] = 'C'
		env['LC_ALL'] = 'C'
		out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
		return out

	try:
		out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
		sha = out.strip().decode('ascii')
	except OSError:
		sha = 'unknown'

	return sha

def get_hash():
	if os.path.exists('.git'):
		sha = get_git_hash()[:7]
	else:
		sha = 'unknown'

	return sha

def write_version_py(version,version_file):
	content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
	sha = get_hash()
	with open(version, 'r') as f:
		SHORT_VERSION = f.read().strip()
	VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])

	version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
	with open(version_file, 'w') as f:
		f.write(version_file_str)
		
def make_cuda_ext(name, module, sources, sources_cuda=None):
	if sources_cuda is None:
		sources_cuda = []
	define_macros = []
	extra_compile_args = {'cxx': []}

	if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
		define_macros += [('WITH_CUDA', None)]
		extension = CUDAExtension
		extra_compile_args['nvcc'] = [
			'-D__CUDA_NO_HALF_OPERATORS__',
			'-D__CUDA_NO_HALF_CONVERSIONS__',
			'-D__CUDA_NO_HALF2_OPERATORS__',
		]
		sources += sources_cuda
	else:
		print(f'Compiling {name} without CUDA')
		extension = CppExtension

	return extension(
		name=f'{module}.{name}',
		sources=[os.path.join(*module.split('.'), p) for p in sources],
		define_macros=define_macros,
		extra_compile_args=extra_compile_args)

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
	
if __name__=="__main__":
	cuda_ext = os.getenv('BASICSR_EXT')
	if cuda_ext == 'True':
		try:
			import torch
			from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
		except ImportError:
			raise ImportError('Unable to import torch - torch is needed to build cuda extensions')

		ext_modules = [
			make_cuda_ext(
				name='deform_conv_ext',
				module='basicsr.ops.dcn',
				sources=['src/deform_conv_ext.cpp'],
				sources_cuda=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu']),
			make_cuda_ext(
				name='fused_act_ext',
				module='basicsr.ops.fused_act',
				sources=['src/fused_bias_act.cpp'],
				sources_cuda=['src/fused_bias_act_kernel.cu']),
			make_cuda_ext(
				name='upfirdn2d_ext',
				module='basicsr.ops.upfirdn2d',
				sources=['src/upfirdn2d.cpp'],
				sources_cuda=['src/upfirdn2d_kernel.cu']),
		]
		setup_kwargs = {'build_ext': BuildExtension}
		setup_requires=['cython', 'numpy','torch']
	else:
		ext_modules = []
		setup_kwargs = {}
		setup_requires=['cython', 'numpy']
	
	write_version_py(version=basicsr_version,version_file=basicsr_version_file)
	write_version_py(version=realesrgan_version,version_file=realesrgan_version_file)
	write_version_py(version=gfpgan_version,version_file=gfpgan_version_file)
	setup(
		name='mokuba_scripts',
		version='1.2.2',
		author='mokuba_attack',
		author_email='mokuatta@nigge.rs',
		package_dir={
			"diffusers":"src/submodules/diffusers",
			"mokuba_scripts":"src/mokuba_scripts",
			"basicsr":"src/submodules/BasicSR/basicsr",
			"realesrgan":"src/submodules/Real-ESRGAN/realesrgan",
			"gfpgan":"src/submodules/GFPGAN/gfpgan",
		},
		packages=find_packages("src") \
			+["diffusers.modular_pipelines.anima"] \
			+find_packages("src/submodules/BasicSR",exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')) \
			+find_packages("src/submodules/Real-ESRGAN",exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')) \
			+find_packages("src/submodules/GFPGAN",exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
		include_package_data=True,
		description='This is a script that I use when I create images by diffusers.',
		long_description=open('README.md').read(),
		long_description_content_type='text/markdown',
		license='BSD-3-Clause',
		classifiers=[
			'License :: OSI Approved :: BSD License',
			'Programming Language :: Python :: 3.12',
			'Programming Language :: Python :: 3.13',
		],
		install_requires=[
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
			"numpy>=1.17",
			"Pillow",
			"addict",
			"future",
			"lmdb",
			"opencv-python",
			"pyyaml",
			"scikit-image",
			"scipy",
			"tb-nightly",
			"tqdm",
			"yapf",
			"facexlib>=0.2.5",
			"diffusers>=0.40.0",
		],
		extras_require={
			"cuda":extras_need(w="cuda"),
			"xpu":extras_need(w="xpu"),
			"cpu":extras_need(w="cpu"),
		},
		entry_points={
			'console_scripts': [
				'mokugui = mokuba_scripts.mokugui:main',
				'mokupatch = mokuba_scripts.mokupatch:main',
			]
		},
		setup_requires=setup_requires,
		ext_modules=ext_modules,
		zip_safe=False,
		cmdclass=setup_kwargs,
	)
