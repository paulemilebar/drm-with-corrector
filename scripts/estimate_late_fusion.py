"""Independent corrected DRMNet trajectories, followed by one-shot late fusion.

All objects must share the same illumination and camera coordinate frame.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_img", type=Path, nargs="+", required=True, help="One HDR image per object, in a common camera frame")
    parser.add_argument("--input_normal", type=Path, nargs="+", required=True, help="Matching normal maps (.npy), in the same order")
    parser.add_argument("--input_mask", type=Path, nargs="+", help="Optional matching masks (.png); otherwise use valid normals")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full corrective checkpoint; vanilla checkpoint is also accepted for the zero-correction baseline")
    parser.add_argument("--drmnet_base_path", type=Path, default=Path("configs/correction/eval.yaml"))
    parser.add_argument("--obsnet_base_path", type=Path, default=Path("configs/obsnet/eval_obsnet.yaml"))
    parser.add_argument("--corrections_per_step", type=int, help="Override the YAML: 0 for baseline, 1 or 2 for correction")
    parser.add_argument("--erode_kernel_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/corrective_late_fusion"))
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    count = len(args.input_img)
    if len(args.input_normal) != count or (args.input_mask is not None and len(args.input_mask) != count):
        parser.error("Provide the same number of images, normal maps and optional masks")
    if args.corrections_per_step is not None and args.corrections_per_step < 0:
        parser.error("corrections_per_step must be nonnegative")
    for path in args.input_img + args.input_normal + (args.input_mask or []) + [args.checkpoint]:
        if not path.is_file():
            parser.error(f"File not found: {path}")

    import mitsuba as mi
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything

    mi.set_variant("cuda_ad_rgb")
    from ldm.util import instantiate_from_config
    from scripts.estimate import estimate
    from utils.file_io import load_exr, load_png, save_exr, save_png
    from utils.tonemap import hdr2ldr

    seed_everything(args.seed)
    config = OmegaConf.load(args.drmnet_base_path)
    config.model.params.ckpt_path = str(args.checkpoint)
    if args.corrections_per_step is not None:
        config.model.params.corrections_per_step = args.corrections_per_step
    model = instantiate_from_config(config.model).cuda().eval()
    model.ds = instantiate_from_config(config.data.params.predict)
    obs_config = OmegaConf.load(args.obsnet_base_path)
    obsnet = instantiate_from_config(obs_config.model).cuda().eval()
    obsnet.ds = instantiate_from_config(obs_config.data.params.predict)
    aggregator = instantiate_from_config(config.aggregation).cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({
        "drmnet": OmegaConf.to_container(config, resolve=True),
        "obsnet": OmegaConf.to_container(obs_config, resolve=True),
        "seed": args.seed,
        "input_img": [str(path.resolve()) for path in args.input_img],
        "input_normal": [str(path.resolve()) for path in args.input_normal],
        "input_mask": [str(path.resolve()) for path in (args.input_mask or [])],
        "erode_kernel_size": args.erode_kernel_size,
    }), args.output_dir / "config.yaml")

    envmaps, materials = [], []
    with torch.no_grad():
        for index, (image_path, normal_path) in enumerate(zip(args.input_img, args.input_normal)):
            # A preceding object's convergence must not change this object's seed.
            seed_everything(args.seed + index)
            image = load_exr(image_path, as_torch=True).cuda()
            normal = torch.from_numpy(np.load(normal_path)).float().cuda()
            if normal.shape != image.shape:
                raise ValueError(f"Object {index}: normals and image must both have shape (H, W, 3)")
            mask = torch.isfinite(normal).all(dim=-1) & (torch.linalg.norm(normal, dim=-1) > 0.5)
            if args.input_mask is not None:
                supplied_mask = load_png(args.input_mask[index], as_torch=True).cuda()
                if supplied_mask.ndim == 3:
                    supplied_mask = supplied_mask[..., 0]
                if supplied_mask.shape != mask.shape:
                    raise ValueError(f"Object {index}: mask dimensions do not match the image")
                mask &= supplied_mask > 0.5
            if not mask.any():
                raise ValueError(f"Object {index}: no valid foreground pixels")

            # Reuse ObsNet, normalization, the inherited sampler and inverse
            # intensity normalization from the official estimate() function.
            refmap, material = estimate(model, obsnet, image, normal, mask,
                                       tag=f"object_{index}", erode_kernel_size=args.erode_kernel_size)
            envmap = model.r0toenvmap(refmap[None], (model.image_size, 2 * model.image_size))[0]
            if not torch.isfinite(material).all() or not torch.isfinite(envmap).all():
                raise RuntimeError(f"Object {index}: nonfinite prediction; check sampler convergence before fusion")
            envmap = envmap.clamp_min(0)
            save_exr(args.output_dir / f"env_agent{index}.exr", envmap)
            save_png(args.output_dir / f"env_agent{index}.png", hdr2ldr(envmap.cpu().numpy()))
            envmaps.append(envmap.permute(2, 0, 1))
            materials.append(material)

        materials = torch.stack(materials)
        # Exactly one aggregation call, after every independent trajectory.
        fused = aggregator(torch.stack(envmaps)[None], materials[None])[0]
        if not torch.isfinite(fused).all():
            raise RuntimeError("Nonfinite late-fusion result")
        save_exr(args.output_dir / "env_fused.exr", fused, channel_first=True)
        save_png(args.output_dir / "env_fused.png", hdr2ldr(fused.permute(1, 2, 0).cpu().numpy()))
        torch.save(materials.cpu(), args.output_dir / "zK.pt")
    print(f"Fused {count} objects; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
