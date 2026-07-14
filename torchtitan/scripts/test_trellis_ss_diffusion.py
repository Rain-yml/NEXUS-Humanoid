import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import glob
import json
import numpy as np
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler
from torchtitan.experiments.trellis.pipelines.scheduler import FlowMatchEulerDiscreteScheduler
from torchtitan.experiments.trellis.vae.structure_vae_8x import SparseStructureDecoder
from torchtitan.experiments.trellis.vae.vecset_structure_vae import SparseStructureDecoder as SparseStructureVecsetDecoder
from torchtitan.experiments.trellis.model.transformer import TrellisTransformer3DModel
from torchtitan.experiments.trellis.pipelines.structure import TrellisStructurePipeline
from torchtitan.experiments.trellis.vae.vis import visualize_structure
from torchtitan.experiments.trellis.image_encoder import DINOv2ImageEncoderWithoutPooler, SigLIP2ImageEncoder, DINOv2ImageEncoder, DINOv3ImageEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", type=str, default="/mnt/pfs/users/caoyanpei/data/test_images/v3.0/captioned_segmented")
parser.add_argument("--decoder-ckpt", type=str, default="/mnt/pfs/users/guoyuanchen/torchtitan/ckpts/ss_decoder_8x_0625.ckpt")
parser.add_argument("--transformer-ckpt", type=str, required=True)
parser.add_argument("--transformer-refiner-ckpt", type=str, default=None)
parser.add_argument("--condition-type", type=str, required=True, choices=["single", "single_dino", "single_dinov3hp", "dual_tc", "dual_cc", "dual_cc_plus"], default="dual_cc")
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--num-inference-steps", type=int, default=20)
parser.add_argument("--guidance-scale", type=float, default=5)
parser.add_argument("--shift", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--normalize-stats", type=str, default=None)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--stochastic-start", type=float, default=1.0)
parser.add_argument("--guidance-start-factor", type=float, default=1.0)
parser.add_argument("--refine-start", type=float, default=None)
parser.add_argument("--noise-std", type=float, default=1.0)
parser.add_argument("--image-size", type=int, default=518)
parser.add_argument("--image-2-size", type=int, default=448)
parser.add_argument('--scheduler-type', type=str, default='euler', choices=['euler', 'dpmpp2msde'])
parser.add_argument('--use-vecset', action='store_true', default=False)


args = parser.parse_args()


# save inference parameters
os.makedirs(args.save_dir, exist_ok=True)
# Save args as json
args_dict = vars(args)
with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
    json.dump(args_dict, f, indent=2)

if args.use_vecset:
    dec = SparseStructureVecsetDecoder(
        out_channels=1,
        channels=[128, 32, 8],
        latent_channels=16,
        num_res_blocks=2,
        num_res_blocks_middle=2,
        norm_type="layer",
        use_fp16=False,
        initialize_type="xavier",
        perceiver_cfg={
            "perceiver_type": "latent2voxel",
            "model_channels": 768,
            "voxel_channels": 128,
            "num_blocks": 8,
            "num_heads": 12,
            "num_heads_channels": 64,
            "mlp_ratio": 4,
            "resolution": 32,
            "num_tokens": [4096],
            "use_checkpoint": False,
            "pe_mode": "ape",
            "qk_rms_norm": True,
            "qk_rms_norm_cross": True,
        },
    )
    dec.load_state_dict(torch.load(args.decoder_ckpt))
    dec.to(device='cuda', dtype=torch.float32).eval()
    dec.dtype = torch.float32

    class DecoderWrapper:
        def __init__(self, dec):
            self.dec = dec
            self.dtype = dec.dtype
            self.latents_mean = dec.latents_mean
            self.latents_std = dec.latents_std
            self.scale_factor = dec.scale_factor
        
        def __call__(self, x):
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            return self.dec(x, 32)
    dec = DecoderWrapper(dec)    
else:
    dec = SparseStructureDecoder(
        out_channels=1,
        channels=[1024, 512, 128, 32],
        latent_channels=32,
        num_res_blocks=2,
        num_res_blocks_middle=2,
        norm_type="layer",
        use_fp16=False
    )
    dec.load_state_dict(torch.load(args.decoder_ckpt))
    dec.to(device='cuda', dtype=torch.bfloat16).eval()
    dec.dtype = torch.bfloat16


if args.normalize:
    normalize_stats_file = args.normalize_stats if args.normalize_stats is not None else os.path.join(os.path.dirname(args.decoder_ckpt), os.path.basename(args.decoder_ckpt).split('.')[0] + '.json')
    normalize_stats = json.load(open(normalize_stats_file))
    dec.latents_mean = normalize_stats['mean']
    dec.latents_std = normalize_stats['std']


transformer_kwargs = dict(
    patch_size=(1, 1, 1),
    num_attention_heads=28,
    attention_head_dim=128,
    in_channels=16 if args.use_vecset else 32,
    out_channels=16 if args.use_vecset else 32,
    ffn_dim=18944,
    num_layers=28,
    norm_type="rms_norm",
    qk_norm="rms_norm_across_heads",
    attn_backend="flash_attn_3",
    max_seq_len_1d=16,
    pos_embed_type="learnable" if args.use_vecset else "rope",
    use_rope4d=True,
    rope_4d_xyz_percentage=0.75,
)

if args.condition_type == "single":
    transformer_kwargs["image_dim"] = 1152
elif args.condition_type == "single_dino":
    transformer_kwargs["image_dim"] = 1024
elif args.condition_type == "single_dinov3hp":
    transformer_kwargs["image_dim"] = 1280
elif args.condition_type == "dual_tc":
    transformer_kwargs["image_dim"] = 1152
    transformer_kwargs["image_dim_2"] = 1024
elif args.condition_type == "dual_cc":
    transformer_kwargs["image_dim"] = 2176
elif args.condition_type == "dual_cc_plus":
    transformer_kwargs["image_dim"] = 1024
    transformer_kwargs["image_dim_2"] = 1152
    transformer_kwargs["image_embed_2_concat_along"] = "channel"

transformer = TrellisTransformer3DModel(**transformer_kwargs)

# FIXME: must in this order to set up position embeddings in correct dtype
states = {'model': transformer.state_dict()}
transformer.to(device='cuda', dtype=torch.bfloat16)
transformer.init_weights()
torch.distributed.checkpoint.load(states, checkpoint_id=args.transformer_ckpt)
transformer.load_state_dict(states['model'])

if args.transformer_refiner_ckpt is not None:
    transformer_refiner_kwargs = transformer_kwargs.copy()
    transformer_refiner_kwargs.update(
        num_attention_heads=12,
        attention_head_dim=128,
        ffn_dim=8960,
        num_layers=30,
    )
    transformer_refiner = TrellisTransformer3DModel(**transformer_refiner_kwargs)
    states = {'model': transformer_refiner.state_dict()}
    transformer_refiner.to(device='cuda', dtype=torch.bfloat16)
    transformer_refiner.init_weights()
    torch.distributed.checkpoint.load(states, checkpoint_id=args.transformer_refiner_ckpt)
    transformer_refiner.load_state_dict(states['model'])
else:
    transformer_refiner = None

# compile transformer blocks
for block_id, transformer_block in transformer.blocks.named_children():
    transformer_block = torch.compile(transformer_block, fullgraph=True)
    transformer.blocks.register_module(block_id, transformer_block)

if args.condition_type == "single_dino" or args.condition_type == "dual_cc_plus":
    image_encoder = DINOv2ImageEncoder(
        model_name='facebook/dinov2-with-registers-large',
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)
elif args.condition_type == "single_dinov3hp":
    image_encoder = DINOv3ImageEncoder(
        model_name="/mnt/pfs/share/pretrained_model/DINOv3/dinov3-vith16plus-pretrain-lvd1689m/",
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)
else:
    image_encoder = SigLIP2ImageEncoder(
        model_name='google/siglip2-so400m-patch16-512',
        return_the_nth_hidden_states=20
    ).to(device='cuda', dtype=torch.bfloat16)

if args.condition_type == "dual_tc":
    image_encoder_2 = DINOv2ImageEncoder(
        model_name='facebook/dinov2-with-registers-large',
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)
elif args.condition_type == "dual_cc_plus":
    image_encoder_2 = SigLIP2ImageEncoder(
        model_name='google/siglip2-so400m-patch16-512',
        return_the_nth_hidden_states=20
    ).to(device='cuda', dtype=torch.bfloat16)
else:
    image_encoder_2 = DINOv2ImageEncoderWithoutPooler(
        model_name='facebook/dinov2-large',
        return_the_nth_hidden_states=-1
    ).to(device='cuda', dtype=torch.bfloat16)

def build_scheduler(scheduler_type, shift, noise_scale):
    if scheduler_type == 'euler':
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, stochastic_sampling=False, noise_scale=noise_scale)
    elif scheduler_type == 'dpmpp2msde':
        assert noise_scale == 1
        return DPMSolverMultistepScheduler(num_train_timesteps=1000, prediction_type='flow_prediction', flow_shift=shift, use_flow_sigmas=True, algorithm_type='sde-dpmsolver++')
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

pipeline = TrellisStructurePipeline(
    image_encoder=image_encoder,
    image_encoder_2=None if "single" in args.condition_type else image_encoder_2,
    transformer=transformer,
    transformer_refiner=transformer_refiner,
    vae=dec,
    scheduler=build_scheduler(args.scheduler_type, shift=args.shift, noise_scale=args.noise_std)
)


image_fns = glob.glob(f'{args.image_dir}/*.webp')
for image_fn in image_fns:
    uuid = os.path.basename(image_fn).replace('.webp', '')

    ss_save_path = os.path.join(args.save_dir, f"{uuid}_ss.npy")
    if os.path.exists(ss_save_path):
        continue

    image = Image.open(image_fn).convert('RGBA')
    # add gray background
    background = Image.new('RGBA', image.size, (128, 128, 128, 255))
    background.paste(image, (0, 0), image)
    image = background.convert('RGB').resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
    image_empty = Image.new('RGB', image.size, (0, 0, 0))

    # save input image
    image.save(os.path.join(args.save_dir, f"{uuid}_input.webp"))
    size = 128
    shapes = pipeline(
        image=image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS),
        image_2=None if "single" in args.condition_type else image.resize((args.image_2_size, args.image_2_size), Image.Resampling.LANCZOS),
        depth=size, height=size, width=size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        concat_hidden_states_along_channel=args.condition_type == "dual_cc",
        noise_std=args.noise_std,
        refine_start=args.refine_start,
        stochastic_range=None if not args.stochastic else (args.stochastic_start, 0.0),
        guidance_range=(args.guidance_start_factor, 1.0),
        generator=torch.Generator(device='cuda').manual_seed(args.seed)
    ).shapes
    vis = visualize_structure(shapes)
    vis[0].save(os.path.join(args.save_dir, f"{uuid}_ss.webp"), lossless=True)
    voxel_grid = (shapes[0,0] > 0).cpu().numpy()

    N = voxel_grid.shape[0]
    packed = voxel_grid.reshape(N//2, 2, N//2, 2, N//2, 2).transpose((0, 2, 4, 1, 3, 5)).reshape(N//2, N//2, N//2, 8)
    packed = np.packbits(packed, axis=-1)[...,0]
    np.save(ss_save_path, packed)
