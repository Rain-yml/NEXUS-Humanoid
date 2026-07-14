import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ['SPARSE_BACKEND'] = 'torchsparse'
import argparse
import shutil
import glob
import json
import time
import random
random.seed(int(time.time()))
import traceback
from PIL import Image
import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

from diffusers import FlowMatchEulerDiscreteScheduler
from torchtitan.experiments.trellis.vae.slat_vae_8x import SLatMeshDecoder
from torchtitan.experiments.trellis.vae.sparse import SparseTensor
from torchtitan.experiments.trellis.vae.vis import visualize_mesh
from torchtitan.experiments.trellis.model.sparse_transformer import TrellisSparseTransformer3DModel
from torchtitan.experiments.trellis.pipelines.slat import TrellisSLATPipeline
from torchtitan.experiments.trellis.image_encoder import DINOv2ImageEncoderWithoutPooler, SigLIP2ImageEncoder, DINOv2ImageEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--decoder-ckpt", type=str, default="/mnt/pfs/users/guoyuanchen/torchtitan/ckpts/slat_decoder_8x_0703.ckpt")
parser.add_argument("--transformer-ckpt", type=str, required=True)
parser.add_argument("--condition-type", type=str, required=True, choices=["single", "single_dino", "dual_cc", "dual_cc_plus"], default="dual_cc")
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--structure-dir", type=str, required=True)
parser.add_argument("--num-inference-steps", type=int, default=20)
parser.add_argument("--guidance-scale", type=float, default=5)
parser.add_argument("--shift", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--normalize-stats", type=str, default=None)
parser.add_argument("--image-size", type=int, default=512)
parser.add_argument("--image-2-size", type=int, default=448)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--hang", action="store_true", default=False)

args = parser.parse_args()

# save inference parameters
os.makedirs(args.save_dir, exist_ok=True)
# Save args as json
args_dict = vars(args)
with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
    json.dump(args_dict, f, indent=2)



dec = SLatMeshDecoder(
    resolution=128,
    model_channels=1024,
    latent_channels=32,
    num_blocks=16,
    num_heads=16,
    num_head_channels=64,
    mlp_ratio=4,
    upsample_layer=3,
    attn_mode="swin",
    window_size=8,
    pe_mode="ape",
    pe_base_resolution=64,
    use_fp16=False,
    use_checkpoint=True,
    qk_rms_norm=True if "0703" in args.decoder_ckpt else False,
    pruning=False,
    use_sparse_flexicube=True,
    use_sparse_sparse_flexicube=False,
    pixel_shuffle=True,
    ln_affine=True,
    representation_config={
      'use_color': False,
      'sdf_bias': 0,
    }
)
dec.load_state_dict(torch.load(args.decoder_ckpt))
dec.convert_to_fp16()
dec.to(device='cuda').eval()


normalize_stats_file = args.normalize_stats if args.normalize_stats is not None else os.path.join(os.path.dirname(args.decoder_ckpt), os.path.basename(args.decoder_ckpt).split('.')[0] + '.json')
normalize_stats = json.load(open(normalize_stats_file))
normalization_mean = torch.tensor(normalize_stats['mean'])
normalization_std = torch.tensor(normalize_stats['std'])


transformer_kwargs = dict(
    patch_size=(2, 2, 2),
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=32,
    out_channels=32,
    ffn_dim=16384,
    num_layers=48,
    norm_type="rms_norm",
    qk_norm="rms_norm_across_heads",
    attn_backend="flash_attn_3",
    max_seq_len_1d=32,
    pos_embed_type="rope",
    use_rope4d=True,
    rope_4d_xyz_percentage=0.75,    
)

if args.condition_type == "single":
    transformer_kwargs["image_dim"] = 1152
elif args.condition_type == "single_dino":
    transformer_kwargs["image_dim"] = 1024
elif args.condition_type == "dual_cc":
    transformer_kwargs["image_dim"] = 2176
elif args.condition_type == "dual_cc_plus":
    transformer_kwargs["image_dim"] = 1024
    transformer_kwargs["image_dim_2"] = 1152
    transformer_kwargs["image_embed_2_concat_along"] = "channel"    

transformer = TrellisSparseTransformer3DModel(**transformer_kwargs)

# FIXME: must in this order to set up position embeddings in correct dtype
states = {'model': transformer.state_dict()}
transformer.to(device='cuda', dtype=torch.bfloat16)
transformer.init_weights()
torch.distributed.checkpoint.load(states, checkpoint_id=args.transformer_ckpt)
transformer.load_state_dict(states['model'])

# compile transformer blocks
for block_id, transformer_block in transformer.blocks.named_children():
    transformer_block = torch.compile(transformer_block, fullgraph=True)
    transformer.blocks.register_module(block_id, transformer_block)

if args.condition_type == "single_dino" or args.condition_type == "dual_cc_plus":
    image_encoder = DINOv2ImageEncoder(
        model_name='facebook/dinov2-with-registers-large',
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)
else:
    image_encoder = SigLIP2ImageEncoder(
        model_name='google/siglip2-so400m-patch16-512',
        return_the_nth_hidden_states=20
    ).to(device='cuda', dtype=torch.bfloat16)

if args.condition_type == "dual_cc_plus":
    image_encoder_2 = SigLIP2ImageEncoder(
        model_name='google/siglip2-so400m-patch16-512',
        return_the_nth_hidden_states=20
    ).to(device='cuda', dtype=torch.bfloat16)
else:
    image_encoder_2 = DINOv2ImageEncoderWithoutPooler(
        model_name='facebook/dinov2-large',
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)

scheduler_kwargs = {}
if args.stochastic:
    scheduler_kwargs["stochastic_sampling"] = True
pipeline = TrellisSLATPipeline(
    image_encoder=image_encoder,
    image_encoder_2=None if "single" in args.condition_type else image_encoder_2,
    transformer=transformer,
    vae=None,
    scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.shift, **scheduler_kwargs)
)


while True:
    processed = 0
    for ss_path in glob.glob(os.path.join(args.structure_dir, "*_ss.npy")):
        try:
            name = os.path.basename(ss_path).split('_ss')[0]

            vis_save_path = os.path.join(args.save_dir, f"{name}_slat.webp")
            if os.path.exists(vis_save_path):
                continue

            image_fn = os.path.join(args.structure_dir, f"{name}_input.webp")
            image = Image.open(image_fn)
            shutil.copyfile(image_fn, os.path.join(args.save_dir, f"{name}_input.webp"))
            ss_vis_path = ss_path.replace('.npy', '.webp')
            shutil.copyfile(ss_vis_path, os.path.join(args.save_dir, os.path.basename(ss_vis_path)))
            
            voxel = np.load(ss_path)
            if voxel.dtype != bool:
                voxel = np.unpackbits(voxel[...,None]).reshape(64, 64, 64, 2, 2, 2).transpose((0, 3, 1, 4, 2, 5)).reshape(128, 128, 128)
                voxel = voxel.reshape(64, 2, 64, 2, 64, 2).max(axis=(1, 3, 5))
            coords = torch.from_numpy(np.stack(np.nonzero(voxel)).T).long()
            latents = pipeline(
                image=image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS),
                coords=coords,
                image_2=None if "single" in args.condition_type else image.resize((args.image_2_size, args.image_2_size), Image.Resampling.LANCZOS),
                num_inference_steps=args.num_inference_steps,               
                num_shapes_per_prompt=1,
                guidance_scale=args.guidance_scale,
                concat_hidden_states_along_channel=args.condition_type == "dual_cc",
                generator=torch.Generator(device='cuda').manual_seed(args.seed),
                output_type="latent"
            ).shapes
            latent = latents[0]
            latent = latent * normalization_std[None].to(latent) + normalization_mean[None].to(latent)
            torch.save({'coords': coords, 'latents': latent}, os.path.join(args.save_dir, f"{name}_slat.pt"))
            bcoords = torch.cat([torch.full_like(coords[:,0:1], 0), coords], dim=-1).cuda()
            sparse_tensors = SparseTensor(latent.half(), bcoords.int())
            mesh = dec(sparse_tensors, resolution=64, upsample_steps=3, chunk_wise=True)
            vis = visualize_mesh(mesh)[0]
            vis.save(vis_save_path, lossless=True)

            processed += 1
        except Exception as e:
            # prevent potential OOM
            print(traceback.format_exc())
            torch.cuda.empty_cache()

    if processed == 0:
        if args.hang:
            time.sleep(5)
        else:
            break
