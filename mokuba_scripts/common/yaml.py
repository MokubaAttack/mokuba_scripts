import json
import torch
from accelerate.utils import (
	is_hpu_available,
	is_mlu_available,
	is_musa_available,
	is_neuron_available,
	is_npu_available,
	is_sdaa_available,
	is_xpu_available,
)

def mk_yaml(n=0):
	sd={
		"compute_environment": "LOCAL_MACHINE",
		"debug": False,
		"distributed_type": "NO",
		"downcast_bf16": False,
		"enable_cpu_affinity": False,
		"machine_rank": 0,
		"main_training_function": "main",
		"mixed_precision": "no",
		"num_machines": 1,
		"num_processes": 1,
		"rdzv_backend": "static",
		"same_network": False,
		"tpu_use_cluster": False,
		"tpu_use_sudo": False,
		"use_cpu": False
	}

	if is_mlu_available():
		n2 = torch.mlu.device_count()
		t2="MULTI_MLU"
	if is_sdaa_available():
		n2 = torch.sdaa.device_count()
		t2="MULTI_SDAA"
	elif is_musa_available():
		n2 = torch.musa.device_count()
		t2="MULTI_MUSA"
	elif is_hpu_available():
		n2 = torch.hpu.device_count()
		t2="MULTI_HPU"
	elif torch.cuda.is_available():
		n2 = torch.cuda.device_count()
		t2="MULTI_GPU"
	elif is_xpu_available():
		n2 = torch.xpu.device_count()
		t2="MULTI_XPU"
	elif is_npu_available():
		n2 = torch.npu.device_count()
		t2="MULTI_NPU"
	elif is_neuron_available():
		n2 = torch.neuron.device_count()
		t2="MULTI_NEURON"
	else:
		n2 = 0

	if n>n2:
		n=n2
	if n==0:
		sd["use_cpu"] = True
		sd["num_processes"] = 1
		sd["distributed_type"] = "NO"
	elif n==1:
		sd["use_cpu"] = False
		sd["num_processes"] = 1
		sd["distributed_type"] = "NO"
	else:
		sd["distributed_type"] = t2
		sd["use_cpu"] = False
		sd["num_processes"] = 2

	f=open("pipecache/process_params.yaml","w")
	json.dump(sd,f,indent=2)
	f.close()
	return "pipecache/process_params.yaml"