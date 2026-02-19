"""Render Gaussian Splatting videos from trajectory files.

Usage:
    # Render trajectory to video
    python render_trajectory.py --traj-file traj.npz --ckpt ckpt.pt -o video.mp4

    # Render with an upward camera offset
    python render_trajectory.py --traj-file traj.npz --ckpt ckpt.pt -o video.mp4 --up-offset 0.1

    # Save modified trajectory without rendering
    python render_trajectory.py --traj-file traj.npz --data-dir ./data -o out.npz --save-traj-only
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json
import os

import numpy as np
import torch
import tqdm
import imageio

from datasets.colmap import Parser
from gsplat.rendering import rasterization
from colmap_loader import setup_image_directory
from generate_trajectory import offset_trajectory
from trajectory_data import TrajectoryData


# ── Coordinate helpers ────────────────────────────────────────────────────────


def _look_at_rotation(
    camera_pos: np.ndarray,
    target_xy: np.ndarray,
    pitch_deg: float = -10.0,
) -> np.ndarray:
    """
    Compute a 3x3 camera-to-world rotation for a camera at camera_pos looking
    toward target_xy.

    Args:
        camera_pos: Camera world position [3,].
        target_xy: Target position [2,] or [3,] (only XY used for azimuth).
        pitch_deg: Pitch in degrees (positive = looking up).

    Returns:
        3x3 rotation matrix (camera-to-world).
    """
    azimuth = target_xy[:2] - camera_pos[:2]
    az_len = np.linalg.norm(azimuth)
    azimuth = np.array([1.0, 0.0]) if az_len < 1e-8 else azimuth / az_len

    pitch_rad = np.deg2rad(pitch_deg)
    forward = np.array(
        [azimuth[0] * np.cos(pitch_rad), azimuth[1] * np.cos(pitch_rad), np.sin(pitch_rad)]
    )

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    r_len = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if r_len < 1e-8 else right / r_len

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    return np.column_stack([right, up, forward])


def _build_trajectory_transforms(
    positions: np.ndarray,
    look_at_target: np.ndarray,
    pitch_deg: float = -10.0,
) -> np.ndarray:
    """
    Build [N, 4, 4] camera-to-world transforms for a look-at trajectory.

    Args:
        positions: Camera positions [N, 3].
        look_at_target: Target position [2,] or [3,].
        pitch_deg: Pitch in degrees (positive = looking up).

    Returns:
        [N, 4, 4] array of camera-to-world matrices.
    """
    N = len(positions)
    Ts = np.zeros((N, 4, 4), dtype=np.float64)
    for i, pos in enumerate(positions):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _look_at_rotation(pos, look_at_target, pitch_deg)
        T[:3, 3] = pos
        Ts[i] = T
    return Ts


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """
    Apply an SE(3) matrix to an array of camera-to-world matrices.

    Args:
        matrix: [4, 4] SE(3) matrix.
        camtoworlds: [N, 4, 4] camera-to-world matrices.

    Returns:
        [N, 4, 4] transformed matrices (rotations renormalised).
    """
    assert matrix.shape == (4, 4)
    assert camtoworlds.ndim == 3 and camtoworlds.shape[1:] == (4, 4)
    result = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(result[:, 0, :3], axis=1)
    result[:, :3, :3] /= scaling[:, None, None]
    return result


# ── Trajectory loaders ────────────────────────────────────────────────────────


def load_trajectory_from_json(
    json_path: Path,
    parser_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load a trajectory from a JSON file produced by an external tool.

    Expected JSON keys:
        - interpolated_trajectory: list of [x, y, z] positions
        - look_at_target: [x, y, z] target point
        - pitch_deg: camera pitch angle (optional, default -10)

    Args:
        json_path: Path to the JSON file.
        parser_transform: Optional [4, 4] normalisation transform from Parser.

    Returns:
        [N, 4, 4] camera-to-world matrices.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    positions = np.array(data["interpolated_trajectory"])
    mean_xy = positions[:, :2].mean(axis=0)
    positions[:, :2] += 0.2 * (mean_xy - positions[:, :2])
    positions[:, 2] = 0.8

    look_at = np.array(data["look_at_target"])
    pitch = float(data.get("pitch_deg", -10.0))

    print(f"  Loaded {len(positions)} positions from JSON")
    print(f"  Look-at: {look_at}, pitch: {pitch}°")

    transforms = _build_trajectory_transforms(positions, look_at, pitch)
    if parser_transform is not None:
        transforms = transform_cameras(parser_transform, transforms)
    return transforms


# ── Checkpoint loading ────────────────────────────────────────────────────────


def load_checkpoint(
    ckpt_path: Union[str, Path, List[Union[str, Path]]],
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, int]:
    """
    Load splat parameters from one or more checkpoint files.

    For distributed training, pass all rank checkpoint paths to concatenate shards.

    Args:
        ckpt_path: Path or list of paths to checkpoint file(s).
        device: PyTorch device string.

    Returns:
        Tuple of (splats ParameterDict, training step number).
    """
    paths: List[Union[str, Path]] = (
        [ckpt_path] if isinstance(ckpt_path, (str, Path)) else list(ckpt_path)
    )
    ckpts = [torch.load(p, map_location=device, weights_only=False) for p in paths]

    splats = torch.nn.ParameterDict()
    for k in ckpts[0]["splats"]:
        splats[k] = torch.nn.Parameter(
            torch.cat([c["splats"][k] for c in ckpts]).to(device)
        )
    return splats, int(ckpts[0].get("step", 0))


# ── Rasterisation ─────────────────────────────────────────────────────────────


def rasterize_splats(
    splats: torch.nn.ParameterDict,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree: int = 3,
    render_mode: str = "RGB+ED",
    camera_model: str = "pinhole",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Rasterize splats for the given camera poses.

    Args:
        splats: ParameterDict with splat parameters.
        camtoworlds: [B, 4, 4] camera-to-world matrices.
        Ks: [B, 3, 3] camera intrinsics.
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        sh_degree: Spherical harmonics degree.
        render_mode: Render output channels ("RGB", "RGB+ED", etc.).
        camera_model: Camera model string.

    Returns:
        Tuple of (render colours, render alphas, info dict).
    """
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    return rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=sh_degree,
        render_mode=render_mode,
        camera_model=camera_model,
    )


# ── Image helpers ─────────────────────────────────────────────────────────────


def project_points_to_image(
    points_3d: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world-space points to 2D image coordinates.

    Args:
        points_3d: [N, 3] world-space points.
        camtoworld: [4, 4] camera-to-world matrix.
        K: [3, 3] camera intrinsics.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (2D points [N, 2], boolean valid mask [N,]).
    """
    w2c = np.linalg.inv(camtoworld)
    pts_cam = (w2c[:3, :3] @ points_3d.T + w2c[:3, 3:4]).T
    valid = pts_cam[:, 2] > 0.1
    pts_proj = (K @ pts_cam.T).T
    pts_2d = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-8)
    valid &= (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < width)
    valid &= (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < height)
    return pts_2d, valid


def draw_trajectory_on_image(
    image: np.ndarray,
    trajectory_positions: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 3,
    line_thickness: int = 2,
    subsample: int = 1,
) -> np.ndarray:
    """
    Draw projected trajectory points and connecting lines onto a uint8 image.

    Args:
        image: [H, W, 3] uint8 image.
        trajectory_positions: [N, 3] camera world positions.
        camtoworld: [4, 4] viewpoint camera-to-world matrix.
        K: [3, 3] camera intrinsics.
        color: RGB draw colour.
        point_radius: Radius of drawn points in pixels.
        line_thickness: Thickness of connecting lines in pixels.
        subsample: Draw every Nth point.

    Returns:
        [H, W, 3] image with trajectory overlaid.
    """
    import cv2

    result = image.copy()
    h, w = image.shape[:2]
    pos = trajectory_positions[::subsample]
    pts_2d, valid = project_points_to_image(pos, camtoworld, K, w, h)

    for i in range(len(pos) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(
                result,
                tuple(pts_2d[i].astype(int)),
                tuple(pts_2d[i + 1].astype(int)),
                color,
                line_thickness,
                cv2.LINE_AA,
            )
    for i in range(len(pos)):
        if valid[i]:
            cv2.circle(result, tuple(pts_2d[i].astype(int)), point_radius, color, -1, cv2.LINE_AA)
    return result


# ── Rendering ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def render_trajectory_to_video(
    splats: torch.nn.ParameterDict,
    camtoworlds: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str,
    fps: int = 30,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree: int = 3,
    show_depth: bool = True,
    device: str = "cuda",
    flip: bool = True,
) -> None:
    """
    Render a camera trajectory to an MP4 video (RGB + optional depth side-by-side).

    Args:
        splats: ParameterDict with splat parameters.
        camtoworlds: [N, 4, 4] camera-to-world matrices.
        K: [3, 3] camera intrinsics.
        width: Render width in pixels.
        height: Render height in pixels.
        output_path: Destination MP4 path.
        fps: Video frame rate.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        sh_degree: Spherical harmonics degree.
        show_depth: If True append normalised depth to the right of RGB.
        device: PyTorch device string.
        flip: If True flip frames vertically (COLMAP Y-down → video Y-up).
    """
    c2w_t = torch.from_numpy(camtoworlds).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    mode = "RGB+ED" if show_depth else "RGB"
    with imageio.get_writer(output_path, fps=fps) as writer:
        for i in tqdm.trange(len(c2w_t), desc="Rendering"):
            renders, _, _ = rasterize_splats(
                splats, c2w_t[i : i + 1], K_t[None], width, height,
                near_plane, far_plane, sh_degree, mode,
            )
            if show_depth and renders.shape[-1] == 4:
                rgb = torch.clamp(renders[..., :3], 0.0, 1.0)
                depth = renders[..., 3:4]
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                canvas = torch.cat([rgb, depth.repeat(1, 1, 1, 3)], dim=2)
            else:
                canvas = torch.clamp(renders[..., :3], 0.0, 1.0)
            frame = (canvas.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            if flip:
                frame = np.flip(frame, axis=0)
            writer.append_data(frame)
    print(f"Video saved to {output_path}")


@torch.no_grad()
def render_dual_trajectory_video(
    splats: torch.nn.ParameterDict,
    original_camtoworlds: np.ndarray,
    offset_camtoworlds: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str,
    fps: int = 30,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree: int = 3,
    device: str = "cuda",
    flip: bool = True,
) -> None:
    """
    Render a video with original trajectory (top) and offset trajectory (bottom),
    each showing RGB + depth.

    Args:
        splats: ParameterDict with splat parameters.
        original_camtoworlds: [N, 4, 4] original camera poses.
        offset_camtoworlds: [N, 4, 4] offset camera poses.
        K: [3, 3] camera intrinsics.
        width: Render width in pixels.
        height: Render height in pixels.
        output_path: Destination MP4 path.
        fps: Video frame rate.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        sh_degree: Spherical harmonics degree.
        device: PyTorch device string.
        flip: If True flip frames vertically.
    """
    import cv2

    orig_t = torch.from_numpy(original_camtoworlds).float().to(device)
    offset_t = torch.from_numpy(offset_camtoworlds).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)
    n_frames = min(len(orig_t), len(offset_t))

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    def _render_row(c2w: torch.Tensor) -> torch.Tensor:
        r, _, _ = rasterize_splats(
            splats, c2w[None], K_t[None], width, height,
            near_plane, far_plane, sh_degree, "RGB+ED",
        )
        rgb = torch.clamp(r[..., :3], 0.0, 1.0)
        depth = r[..., 3:4]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return torch.cat([rgb, depth.repeat(1, 1, 1, 3)], dim=2)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for i in tqdm.trange(n_frames, desc="Rendering dual"):
            canvas = torch.cat([_render_row(orig_t[i]), _render_row(offset_t[i])], dim=1)
            frame = (canvas.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Offset", (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if flip:
                frame = np.flip(frame, axis=0)
            writer.append_data(frame)
    print(f"Dual video saved to {output_path}")


# ── Persistence ───────────────────────────────────────────────────────────────


def save_trajectory(
    camtoworlds: np.ndarray,
    output_path: Path,
    intrinsics: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Save a trajectory to .npy or .npz.

    Args:
        camtoworlds: [N, 4, 4] camera-to-world matrices.
        output_path: Destination file path.
        intrinsics: Optional [3, 3] intrinsics to embed in .npz.
        image_size: Optional (width, height) to embed in .npz.
    """
    output_path = Path(output_path)
    if output_path.suffix == ".npz":
        data: Dict[str, np.ndarray] = {"camtoworlds": camtoworlds}
        if intrinsics is not None:
            data["intrinsics"] = intrinsics
        if image_size is not None:
            data["image_size"] = np.array(image_size)
        np.savez(output_path, **data)
    else:
        np.save(output_path, camtoworlds)
    print(f"Saved {len(camtoworlds)} poses to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Gaussian Splatting video from a trajectory file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python render_trajectory.py --traj-file traj.npz --ckpt ckpt.pt -o video.mp4
  python render_trajectory.py --traj-file traj.npz --ckpt ckpt.pt -o video.mp4 --up-offset 0.1
  python render_trajectory.py --traj-file traj.npz --data-dir ./data -o out.npz --save-traj-only
        """,
    )

    parser.add_argument(
        "--traj-file", type=str, required=True,
        help="Trajectory file (.npy, .npz, or .json). Required.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="COLMAP data directory. Required when loading .json or .npy trajectories.",
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None,
        help="Checkpoint file(s). Pass multiple paths for distributed-training shards.",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output file (.mp4 for video, .npy/.npz for trajectory).",
    )
    parser.add_argument("--offset-x", type=float, default=0.0, help="World-space X offset.")
    parser.add_argument("--offset-y", type=float, default=0.0, help="World-space Y offset.")
    parser.add_argument(
        "--offset-z", type=float, default=0.0,
        help="World-space Z offset (up is -Z in gsplat).",
    )
    parser.add_argument(
        "--up-offset", type=float, default=None,
        help="Move camera up by this many metres (shorthand: sets offset-z = -up-offset).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit trajectory length.")
    parser.add_argument("--data-factor", type=int, default=1, help="Image downsample factor.")
    parser.add_argument("--scene-scale", type=float, default=None, help="Override scene scale.")
    parser.add_argument("--sh-degree", type=int, default=3, help="Spherical harmonics degree.")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
    parser.add_argument("--no-depth", action="store_true", help="Render RGB only (no depth).")
    parser.add_argument(
        "--save-traj-only", action="store_true",
        help="Save trajectory file without rendering a video.",
    )
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument(
        "--dual-video", action="store_true",
        help="Render original (top) and offset (bottom) trajectories side by side.",
    )
    parser.add_argument(
        "--flip", action="store_true",
        help="Flip frames vertically (use when video appears upside down).",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(args.output)
    is_video = output_path.suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}

    if is_video and args.ckpt is None and not args.save_traj_only:
        parser.error("--ckpt is required for video rendering.")

    # ── Load trajectory ───────────────────────────────────────────────────────
    traj_path = Path(args.traj_file)
    traj_suffix = traj_path.suffix.lower()

    K: np.ndarray
    img_w: int
    img_h: int
    camtoworlds: np.ndarray
    _scene_scale: float

    if traj_suffix == ".json":
        if args.data_dir is None:
            parser.error("--data-dir is required when loading JSON trajectories.")
        try:
            setup_image_directory(Path(args.data_dir), factor=args.data_factor)
        except Exception as e:
            print(f"Warning: {e}")
        colmap_parser = Parser(
            data_dir=args.data_dir, factor=args.data_factor, normalize=True, test_every=8
        )
        camtoworlds = load_trajectory_from_json(traj_path, colmap_parser.transform)
        _scene_scale = args.scene_scale or colmap_parser.scene_scale * 1.1
        first_cam = list(colmap_parser.Ks_dict.keys())[0]
        K = colmap_parser.Ks_dict[first_cam]
        img_w, img_h = colmap_parser.imsize_dict[first_cam]

    elif traj_suffix == ".npz":
        traj_data = TrajectoryData.load(traj_path)
        camtoworlds = traj_data.camtoworlds
        K = traj_data.intrinsics
        img_w, img_h = traj_data.width, traj_data.height
        _scene_scale = args.scene_scale or 1.0

    else:  # .npy
        if args.data_dir is None:
            parser.error("--data-dir is required when loading .npy trajectories.")
        camtoworlds = np.load(traj_path)
        _scene_scale = args.scene_scale or 1.0
        colmap_parser = Parser(
            data_dir=args.data_dir, factor=args.data_factor, normalize=True, test_every=8
        )
        first_cam = list(colmap_parser.Ks_dict.keys())[0]
        K = colmap_parser.Ks_dict[first_cam]
        img_w, img_h = colmap_parser.imsize_dict[first_cam]

    print(f"Loaded {len(camtoworlds)} poses from {traj_path}")
    print(f"Using scene_scale={_scene_scale:.4f}")

    if args.max_frames is not None and args.max_frames > 0:
        camtoworlds = camtoworlds[: args.max_frames]

    # ── Apply offset ──────────────────────────────────────────────────────────
    offset_z = args.offset_z
    if args.up_offset is not None:
        offset_z = -args.up_offset  # up is -Z in gsplat

    offset = np.array([args.offset_x, args.offset_y, offset_z])
    has_offset = bool(np.any(offset != 0))
    original_camtoworlds = camtoworlds

    if has_offset:
        scaled_offset = offset / _scene_scale
        print(f"Applying offset: world={offset} → scene={scaled_offset}")
        camtoworlds = offset_trajectory(camtoworlds, scaled_offset)

    print(f"Using {len(camtoworlds)} poses")

    # ── Output ────────────────────────────────────────────────────────────────
    if args.save_traj_only or not is_video:
        save_trajectory(camtoworlds, output_path, intrinsics=K, image_size=(img_w, img_h))
        return

    print(f"Loading checkpoint from {args.ckpt}...")
    splats, step = load_checkpoint(args.ckpt, device=device)
    print(f"Loaded step={step}, {len(splats['means'])} Gaussians")

    if args.dual_video and has_offset:
        render_dual_trajectory_video(
            splats=splats,
            original_camtoworlds=original_camtoworlds,
            offset_camtoworlds=camtoworlds,
            K=K,
            width=img_w,
            height=img_h,
            output_path=str(output_path),
            fps=args.fps,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            sh_degree=args.sh_degree,
            device=device,
            flip=args.flip,
        )
    else:
        render_trajectory_to_video(
            splats=splats,
            camtoworlds=camtoworlds,
            K=K,
            width=img_w,
            height=img_h,
            output_path=str(output_path),
            fps=args.fps,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            sh_degree=args.sh_degree,
            show_depth=not args.no_depth,
            device=device,
            flip=args.flip,
        )


if __name__ == "__main__":
    main()
