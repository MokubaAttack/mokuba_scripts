class anima:
	root_map = {
		"x_embedder.proj.1.weight": "patch_embed.proj.weight",
		"t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
		"t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
		"t_embedding_norm.weight": "time_embed.norm.weight",
		"final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
		"final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
		"final_layer.linear.weight": "proj_out.weight",
	}

	block_maps = {
		"adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
		"adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
		"adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
		"adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
		"adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
		"adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
		"self_attn.q_norm.weight": "attn1.norm_q.weight",
		"self_attn.k_norm.weight": "attn1.norm_k.weight",
		"self_attn.q_proj.weight": "attn1.to_q.weight",
		"self_attn.k_proj.weight": "attn1.to_k.weight",
		"self_attn.v_proj.weight": "attn1.to_v.weight",
		"self_attn.output_proj.weight": "attn1.to_out.0.weight",
		"cross_attn.q_norm.weight": "attn2.norm_q.weight",
		"cross_attn.k_norm.weight": "attn2.norm_k.weight",
		"cross_attn.q_proj.weight": "attn2.to_q.weight",
		"cross_attn.k_proj.weight": "attn2.to_k.weight",
		"cross_attn.v_proj.weight": "attn2.to_v.weight",
		"cross_attn.output_proj.weight": "attn2.to_out.0.weight",
		"mlp.layer1.weight": "ff.net.0.proj.weight",
		"mlp.layer2.weight": "ff.net.2.weight",
	}

	vae_keys1={
		"conv1":"quant_conv",
		"conv2":"post_quant_conv",
	}
	vae_keys2={
		"conv1":"conv_in",
		"head.0":"norm_out",
		"head.2":"conv_out",
		"downsamples":"down_blocks",
		"residual.2":"conv1",
		"residual.6":"conv2",
		"residual.0":"norm1",
		"residual.3":"norm2",
		"shortcut":"conv_shortcut",
		"middle.1":"mid_block.attentions.0",
		"middle.0":"mid_block.resnets.0",
		"middle.2":"mid_block.resnets.1",
	}
	vae_keys3={
		"conv1":"conv_in",
		"head.0":"norm_out",
		"head.2":"conv_out",
		"residual.2":"conv1",
		"residual.6":"conv2",
		"residual.0":"norm1",
		"residual.3":"norm2",
		"middle.1":"mid_block.attentions.0",
		"middle.0":"mid_block.resnets.0",
		"middle.2":"mid_block.resnets.1",
		"upsamples.3":"up_blocks.0.upsamplers.0",
		"upsamples.7":"up_blocks.1.upsamplers.0",
		"upsamples.11":"up_blocks.2.upsamplers.0",
		"upsamples.0":"up_blocks.0.resnets.0",
		"upsamples.10":"up_blocks.2.resnets.2",
		"upsamples.12":"up_blocks.3.resnets.0",
		"upsamples.13":"up_blocks.3.resnets.1",
		"upsamples.14":"up_blocks.3.resnets.2",
		"upsamples.1":"up_blocks.0.resnets.1",
		"upsamples.2":"up_blocks.0.resnets.2",
		"upsamples.4":"up_blocks.1.resnets.0",
		"shortcut":"conv_shortcut",
		"upsamples.5":"up_blocks.1.resnets.1",
		"upsamples.6":"up_blocks.1.resnets.2",
		"upsamples.8":"up_blocks.2.resnets.0",
		"upsamples.9":"up_blocks.2.resnets.1",
	}

class sdxl:
	unet_conversion_map={
		"time_embed.0.": "time_embedding.linear_1.",
		"time_embed.2.": "time_embedding.linear_2.",
		"input_blocks.0.0.": "conv_in.",
		"out.0.": "conv_norm_out.",
		"out.2.": "conv_out.",
		"label_emb.0.0.": "add_embedding.linear_1.",
		"label_emb.0.2.": "add_embedding.linear_2.",
	}
	unet_conversion_map_resnet={
		"in_layers.0": "norm1",
		"in_layers.2": "conv1",
		"out_layers.0": "norm2",
		"out_layers.3": "conv2",
		"emb_layers.1": "time_emb_proj",
		"skip_connection": "conv_shortcut",
	}
	unet_conversion_map_layer=[]
	for i in range(3):
		for j in range(2):
			unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".0.","down_blocks."+str(i)+".resnets."+str(j)+".")]
			if i > 0:
				unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".1.","down_blocks."+str(i)+".attentions."+str(j)+".")]

		for j in range(3):
			unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".0.","up_blocks."+str(i)+".resnets."+str(j)+".")]
			if i < 2:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".1.","up_blocks."+str(i)+".attentions."+str(j)+".")]

		if i < 3:
			unet_conversion_map_layer+=[("input_blocks."+str(3 * (i + 1))+".0.op.","down_blocks."+str(i)+".downsamplers.0.conv.")]

			if i==0:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".1.","up_blocks."+str(i)+".upsamplers.0.")]
			else:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".2.","up_blocks."+str(i)+".upsamplers.0.")]
	unet_conversion_map_layer+=[("output_blocks.2.2.conv.","output_blocks.2.1.conv.")]

	unet_conversion_map_layer+=[("middle_block.1.","mid_block.attentions.0.")]
	for j in range(2):
		unet_conversion_map_layer+=[("middle_block."+str(2 * j)+".","mid_block.resnets."+str(j)+".")]
		
	vae_conversion_map={
		"nin_shortcut": "conv_shortcut",
		"norm_out": "conv_norm_out",
		"mid.attn_1.": "mid_block.attentions.0.",
		"mid.block_1.":"mid_block.resnets.0.",
		"mid.block_2.":"mid_block.resnets.1.",
	}
	for i in range(4):
		for j in range(2):
			vae_conversion_map["encoder.down."+str(i)+".block."+str(j)+"."]="encoder.down_blocks."+str(i)+".resnets."+str(j)+"."
	
		if i < 3:
			vae_conversion_map["down."+str(i)+".downsample."]="down_blocks."+str(i)+".downsamplers.0."
			vae_conversion_map["up."+str(3 - i)+".upsample."]="up_blocks."+str(i)+".upsamplers.0."
	
		for j in range(3):
			vae_conversion_map["decoder.up."+str(3 - i)+".block."+str(j)+"."]="decoder.up_blocks."+str(i)+".resnets."+str(j)+"."
	vae_conversion_map_attn={
		"norm.": "group_norm.",
		"q.": "to_q.",
		"k.": "to_k.",
		"v.": "to_v.",
		"proj_out.": "to_out.0.",
	}
	
	textenc_conversion_lst={
		"transformer.resblocks.": "text_model.encoder.layers.",
		".attn":".self_attn",
		"ln_1":"layer_norm1",
		"ln_2":"layer_norm2",
		".c_fc.":".fc1.",
		".c_proj.":".fc2.",
	}
	textenc_root_map={
		"ln_final.": "text_model.final_layer_norm.",
		"token_embedding.weight": "text_model.embeddings.token_embedding.weight",
		"positional_embedding": "text_model.embeddings.position_embedding.weight",
		"text_projection": "text_projection.weight",
	}

class sdkeys:
	unet_conversion_map={
		"time_embed.0.": "time_embedding.linear_1.",
		"time_embed.2.": "time_embedding.linear_2.",
		"input_blocks.0.0.": "conv_in.",
		"out.0.": "conv_norm_out.",
		"out.2.": "conv_out.",
	}
	unet_conversion_map_resnet={
		"in_layers.0": "norm1",
		"in_layers.2": "conv1",
		"out_layers.0": "norm2",
		"out_layers.3": "conv2",
		"emb_layers.1": "time_emb_proj",
		"skip_connection": "conv_shortcut",
	}
	unet_conversion_map_layer=[]
	for i in range(4):
		for j in range(2):
			unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".0.","down_blocks."+str(i)+".resnets."+str(j)+".")]
			if i < 3:
				unet_conversion_map_layer+=[("input_blocks."+str(3 * i + j + 1)+".1.","down_blocks."+str(i)+".attentions."+str(j)+".")]

		for j in range(3):
			unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".0.","up_blocks."+str(i)+".resnets."+str(j)+".")]
			if i > 0:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + j)+".1.","up_blocks."+str(i)+".attentions."+str(j)+".")]

		if i < 3:
			unet_conversion_map_layer+=[("input_blocks."+str(3 * (i + 1))+".0.op.","down_blocks."+str(i)+".downsamplers.0.conv.")]

			if i==0:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".1.","up_blocks."+str(i)+".upsamplers.0.")]
			else:
				unet_conversion_map_layer+=[("output_blocks."+str(3 * i + 2)+".2.","up_blocks."+str(i)+".upsamplers.0.")]

	unet_conversion_map_layer+=[("middle_block.1.","mid_block.attentions.0.")]
	for j in range(2):
		unet_conversion_map_layer+=[("middle_block."+str(2 * j)+".","mid_block.resnets."+str(j)+".")]
