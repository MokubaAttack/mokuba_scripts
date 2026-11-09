import os
import shutil
import json

def save_safe(data_dict,out_path,folder):
	out_dict={}
	out_dict["__metadata__"]={"format":"pt"}
	n=0
	for k in data_dict:
		f=open(folder+"/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l).decode()
		head=json.loads(head)
		out_dict[k]=head[k]
		offsets=out_dict[k]["data_offsets"][1]
		out_dict[k]["data_offsets"][0]=n
		n=n+offsets
		out_dict[k]["data_offsets"][1]=n
		f.close()

	output=open(out_path,"wb")
	out_dict=str(out_dict).replace("'",'"')
	out_dict=out_dict.encode()
	l=len(out_dict).to_bytes(8,byteorder="little")
	output.write(l)
	output.write(out_dict)

	key_count=0
	for k in data_dict:
		f=open(folder+"/"+k+".safetensors","rb")
		l=int.from_bytes(f.read(8),byteorder="little")
		head=f.read(l)
		output.write(f.read())
		f.close()
		os.remove(folder+"/"+k+".safetensors")
	output.close()
	if len(os.listdir(folder)) == 0:
		shutil.rmtree(folder)
