"""
Generate and render camera trajectories for Gaussian Splatting.

This script generates camera trajectories based on COLMAP datasets,
with support for various trajectory types and camera position offsets,
and renders them to MP4 video.

Usage:
    # Render interpolated trajectory with camera moved up 10cm
    python generate_traj.py --data-dir /path/to/data --ckpt /path/to/ckpt.pt --output video.mp4 --up-offset 0.1

    # Render ellipse trajectory
    python generate_traj.py --data-dir /path/to/data --ckpt /path/to/ckpt.pt --output video.mp4 --traj-type ellipse

    # Render using original COLMAP camera poses
    python generate_traj.py --data-dir /path/to/data --ckpt /path/to/ckpt.pt --output video.mp4 --traj-type colmap

    # Render using COLMAP poses with interpolation (smoother video)
    python generate_traj.py --data-dir /path/to/data --ckpt /path/to/ckpt.pt --output video.mp4 --traj-type colmap --colmap-interp 3
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import argparse
import os
import numpy as np
import torch
import tqdm
import imageio

from datasets.colmap import Parser
from gsplat.rendering import rasterization
from trajectory_local import (
    load_trajectory,
    offset_trajectory,
    generate_trajectory_from_parser,
    setup_image_directory,
)                                                                                                               
                       

def _look_at_rotation(
    camera_pos: np.ndarray,
    target_xy: np.ndarray,
    pitch_deg: float = -10.0,
) -> np.ndarray:
    """
    Compute rotation matrix for camera looking towards target_xy (azimuth) with fixed pitch.
    Camera convention: Z points from camera towards target, Y is up.
    Returns 3x3 rotation matrix (camera-to-world).

    Args:
        camera_pos: 3D camera position
        target_xy: 2D or 3D target position (only XY components used for azimuth)
        pitch_deg: Pitch angle in degrees (positive = looking up)
    """
    # Compute azimuth direction from camera XY to target XY
    camera_xy = camera_pos[:2]
    target_2d = target_xy[:2]

    azimuth_dir = target_2d - camera_xy
    azimuth_len = np.linalg.norm(azimuth_dir)

    if azimuth_len < 1e-8:
        # Default to looking along +X if target is at camera position
        azimuth_dir = np.array([1.0, 0.0])
    else:
        azimuth_dir = azimuth_dir / azimuth_len

    # Convert pitch to radians (positive pitch = looking down)
    pitch_rad = np.deg2rad(pitch_deg)

    # Forward direction: azimuth in XY, tilted by pitch (positive = up)
    forward = np.array(
        [
            azimuth_dir[0] * np.cos(pitch_rad),
            azimuth_dir[1] * np.cos(pitch_rad),
            np.sin(pitch_rad),
        ]
    )

    # World up
    world_up = np.array([0.0, 0.0, 1.0])

    # Right = forward x up
    right = np.cross(forward, world_up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-8:
        # Forward is parallel to world up (looking straight up or down)
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len

    # Up = right x forward
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Camera-to-world rotation: columns are right, up, forward
    # Z axis points from camera towards target
    rotation = np.column_stack([right, up, forward])
    return rotation


def _build_trajectory_transforms(
    positions: np.ndarray,
    look_at_target: np.ndarray,
    pitch_deg: float = -10.0,
) -> np.ndarray:
    """
    Build camera-to-world 4x4 transforms for trajectory.
    Each camera is at `positions[i]` and looks at `look_at_target` with `pitch_deg` downward.
    Returns [N, 4, 4] array.
    """
    N = len(positions)
    transforms = np.zeros((N, 4, 4), dtype=np.float64)

    for i, pos in enumerate(positions):
        R = _look_at_rotation(pos, look_at_target, pitch_deg)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        transforms[i] = T

    return transforms


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def load_trajectory_from_json(
    json_path: Path,
    parser_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load trajectory from JSON file (produced by generate_trajectory.py).

    The JSON contains:
        - interpolated_trajectory: list of [x, y, z] positions
        - look_at_target: [x, y, z] target point
        - pitch_deg: camera pitch angle

    Args:
        json_path: Path to trajectory.json file
        parser_transform: 4x4 transform matrix from Parser for normalization.
                         If provided, the trajectory is transformed to match
                         the normalized coordinate system used by gsplat.

    Returns:
        Camera to world transforms [N, 4, 4]
    """
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    positions = np.array(data["interpolated_trajectory"])  # (200, 3)
    mean_xy = positions[:, :2].mean(axis=0)                # (2,)
    # bring the points 0.2 towards mean in XY
    alpha = 0.2  # move 20% of the way toward mean
    positions[:, :2] = positions[:, :2] + alpha * (mean_xy - positions[:, :2])
    positions[:,2] = 0.8
    look_at_target = np.array(data["look_at_target"])
    pitch_deg = data.get("pitch_deg", -10.0)

    print(f"  Loaded {len(positions)} positions from JSON")
    print(f"  Look-at target: {look_at_target}")
    print(f"  Pitch: {pitch_deg} degrees")

    # Build 4x4 transforms from positions and orientation
    transforms = _build_trajectory_transforms(positions, look_at_target, pitch_deg)

    # Apply normalization transform if provided
    if parser_transform is not None:
        print(f"  Applying parser transform for normalization...")
        transforms = transform_cameras(parser_transform, transforms)

    return transforms


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> Tuple[torch.nn.ParameterDict, int]:
    """
    Load splats from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        Tuple of (splats ParameterDict, step number)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    splats = torch.nn.ParameterDict()
    for k, v in ckpt["splats"].items():
        splats[k] = torch.nn.Parameter(v.to(device))
    
    step = ckpt.get("step", 0)
    return splats, step


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
    Rasterize splats for given camera poses.
    """
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
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
    
    return render_colors, render_alphas, info


def project_points_to_image(
    points_3d: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: 3D points [N, 3]
        camtoworld: Camera to world matrix [4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height
    
    Returns:
        Tuple of (2D points [M, 2], valid mask [N,])
    """
    # Transform to camera coordinates
    worldtocam = np.linalg.inv(camtoworld)
    points_cam = (worldtocam[:3, :3] @ points_3d.T + worldtocam[:3, 3:4]).T  # [N, 3]
    
    # Check if points are in front of camera
    valid = points_cam[:, 2] > 0.1
    
    # Project to image
    points_proj = (K @ points_cam.T).T  # [N, 3]
    points_2d = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)  # [N, 2]
    
    # Check if points are within image bounds
    valid &= (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width)
    valid &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    
    return points_2d, valid


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
    Draw a trajectory (camera positions) onto an image.
    
    Args:
        image: Image to draw on [H, W, 3] uint8
        trajectory_positions: Camera positions [N, 3]
        camtoworld: Viewpoint camera to world matrix [4, 4]
        K: Camera intrinsics [3, 3]
        color: RGB color tuple
        point_radius: Radius of trajectory points
        line_thickness: Thickness of connecting lines
        subsample: Draw every Nth point
    
    Returns:
        Image with trajectory drawn [H, W, 3]
    """
    import cv2
    
    result = image.copy()
    height, width = image.shape[:2]
    
    # Subsample trajectory
    positions = trajectory_positions[::subsample]
    
    # Project trajectory points
    points_2d, valid = project_points_to_image(positions, camtoworld, K, width, height)
    
    # Draw lines connecting trajectory points
    for i in range(len(positions) - 1):
        if valid[i] and valid[i + 1]:
            pt1 = tuple(points_2d[i].astype(int))
            pt2 = tuple(points_2d[i + 1].astype(int))
            cv2.line(result, pt1, pt2, color, line_thickness, cv2.LINE_AA)
    
    # Draw points
    for i in range(len(positions)):
        if valid[i]:
            pt = tuple(points_2d[i].astype(int))
            cv2.circle(result, pt, point_radius, color, -1, cv2.LINE_AA)
    
    return result


@torch.no_grad()
def render_trajectory_comparison(
    splats: torch.nn.ParameterDict,
    original_trajectory: np.ndarray,
    offset_trajectory_poses: np.ndarray,
    viewpoint_camtoworld: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    sh_degree: int = 3,
    device: str = "cuda",
    flip: bool = True,
) -> None:
    """
    Render a comparison image showing both trajectories overlaid on the point cloud.
    
    Args:
        splats: ParameterDict with splat parameters
        original_trajectory: Original camera poses [N, 4, 4]
        offset_trajectory_poses: Offset camera poses [N, 4, 4]
        viewpoint_camtoworld: Viewpoint to render from [4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height
        output_path: Path to save the comparison image
        near_plane: Near clipping plane
        far_plane: Far clipping plane
        sh_degree: Spherical harmonics degree
        device: Device to render on
    """
    # Render the scene from the viewpoint
    c2w = torch.from_numpy(viewpoint_camtoworld).float().to(device)[None]
    K_t = torch.from_numpy(K).float().to(device)[None]
    
    renders, _, _ = rasterize_splats(
        splats=splats,
        camtoworlds=c2w,
        Ks=K_t,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=sh_degree,
        render_mode="RGB",
    )
    
    # Convert to numpy image
    image = torch.clamp(renders[0], 0.0, 1.0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    # Extract camera positions from trajectories
    original_positions = original_trajectory[:, :3, 3]  # [N, 3]
    offset_positions = offset_trajectory_poses[:, :3, 3]  # [N, 3]
    
    # Draw original trajectory (blue)
    image = draw_trajectory_on_image(
        image, original_positions, viewpoint_camtoworld, K,
        color=(0, 100, 255),  # Blue (BGR->RGB)
        point_radius=4,
        line_thickness=2,
        subsample=max(1, len(original_positions) // 200),
    )
    
    # Draw offset trajectory (red)
    image = draw_trajectory_on_image(
        image, offset_positions, viewpoint_camtoworld, K,
        color=(255, 50, 50),  # Red
        point_radius=4,
        line_thickness=2,
        subsample=max(1, len(offset_positions) // 200),
    )
    
    # Add legend
    import cv2
    cv2.putText(image, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(image, "Offset", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
    
    # Flip vertically: COLMAP uses Y-down but images expect Y-up
    if flip:
        image = np.flip(image, axis=0)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    imageio.imwrite(output_path, image)
    print(f"Trajectory comparison saved to {output_path}")


@torch.no_grad()
def render_trajectory_comparison_video(
    splats: torch.nn.ParameterDict,
    original_trajectory: np.ndarray,
    offset_trajectory_poses: np.ndarray,
    viewpoint_trajectory: np.ndarray,
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
    Render a video showing the scene with both trajectories overlaid,
    from multiple viewpoints.
    """
    writer = imageio.get_writer(output_path, fps=fps)
    
    original_positions = original_trajectory[:, :3, 3]
    offset_positions = offset_trajectory_poses[:, :3, 3]
    
    K_t = torch.from_numpy(K).float().to(device)
    
    for i in tqdm.trange(len(viewpoint_trajectory), desc="Rendering trajectory comparison"):
        viewpoint = viewpoint_trajectory[i]
        c2w = torch.from_numpy(viewpoint).float().to(device)[None]
        
        renders, _, _ = rasterize_splats(
            splats=splats,
            camtoworlds=c2w,
            Ks=K_t[None],
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode="RGB",
        )
        
        image = torch.clamp(renders[0], 0.0, 1.0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        # Draw trajectories
        image = draw_trajectory_on_image(
            image, original_positions, viewpoint, K,
            color=(0, 100, 255),
            point_radius=3,
            line_thickness=2,
            subsample=max(1, len(original_positions) // 200),
        )
        image = draw_trajectory_on_image(
            image, offset_positions, viewpoint, K,
            color=(255, 50, 50),
            point_radius=3,
            line_thickness=2,
            subsample=max(1, len(offset_positions) // 200),
        )
        
        # Add legend
        import cv2
        cv2.putText(image, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.putText(image, "Offset", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        
        # Flip vertically: COLMAP uses Y-down but video expects Y-up
        if flip:
            image = np.flip(image, axis=0)
        writer.append_data(image)
    
    writer.close()
    print(f"Trajectory comparison video saved to {output_path}")


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
    Render a trajectory to MP4 video.
    """
    camtoworlds_t = torch.from_numpy(camtoworlds).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    render_mode = "RGB+ED" if show_depth else "RGB"
    
    for i in tqdm.trange(len(camtoworlds_t), desc="Rendering trajectory"):
        c2w = camtoworlds_t[i:i+1]
        Ks = K_t[None]
        
        renders, _, _ = rasterize_splats(
            splats=splats,
            camtoworlds=c2w,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode=render_mode,
        )
        
        if show_depth and renders.shape[-1] == 4:
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            canvas = torch.cat([colors, depths.repeat(1, 1, 1, 3)], dim=2)
        else:
            canvas = torch.clamp(renders[..., 0:3], 0.0, 1.0)
        
        frame = canvas.squeeze(0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        # Flip vertically: COLMAP uses Y-down but video expects Y-up
        if flip:
            frame = np.flip(frame, axis=0)
        writer.append_data(frame)
    
    writer.close()
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
    Render a video with two rows:
    - Top row: RGB + Depth from original (unshifted) trajectory
    - Bottom row: RGB + Depth from offset (shifted) trajectory
    
    Args:
        splats: ParameterDict with splat parameters
        original_camtoworlds: Original camera poses [N, 4, 4]
        offset_camtoworlds: Offset camera poses [N, 4, 4]
        K: Camera intrinsics [3, 3]
        width: Image width
        height: Image height
        output_path: Output video path
        fps: Frames per second
        near_plane: Near clipping plane
        far_plane: Far clipping plane
        sh_degree: Spherical harmonics degree
        device: Device to render on
    """
    import cv2
    
    original_t = torch.from_numpy(original_camtoworlds).float().to(device)
    offset_t = torch.from_numpy(offset_camtoworlds).float().to(device)
    K_t = torch.from_numpy(K).float().to(device)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    n_frames = min(len(original_t), len(offset_t))
    
    for i in tqdm.trange(n_frames, desc="Rendering dual trajectory"):
        # Render original trajectory
        c2w_orig = original_t[i:i+1]
        renders_orig, _, _ = rasterize_splats(
            splats=splats,
            camtoworlds=c2w_orig,
            Ks=K_t[None],
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode="RGB+ED",
        )
        
        # Render offset trajectory
        c2w_offset = offset_t[i:i+1]
        renders_offset, _, _ = rasterize_splats(
            splats=splats,
            camtoworlds=c2w_offset,
            Ks=K_t[None],
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode="RGB+ED",
        )
        
        # Process original: RGB + Depth
        colors_orig = torch.clamp(renders_orig[..., 0:3], 0.0, 1.0)
        depths_orig = renders_orig[..., 3:4]
        depths_orig = (depths_orig - depths_orig.min()) / (depths_orig.max() - depths_orig.min() + 1e-8)
        row_orig = torch.cat([colors_orig, depths_orig.repeat(1, 1, 1, 3)], dim=2)
        
        # Process offset: RGB + Depth
        colors_offset = torch.clamp(renders_offset[..., 0:3], 0.0, 1.0)
        depths_offset = renders_offset[..., 3:4]
        depths_offset = (depths_offset - depths_offset.min()) / (depths_offset.max() - depths_offset.min() + 1e-8)
        row_offset = torch.cat([colors_offset, depths_offset.repeat(1, 1, 1, 3)], dim=2)
        
        # Stack rows: original on top, offset on bottom
        canvas = torch.cat([row_orig, row_offset], dim=1)  # [1, 2*H, 2*W, 3]
        
        frame = canvas.squeeze(0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        # Add labels
        label_height = 30
        cv2.putText(frame, "Original", (10, label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Offset", (10, height + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Flip vertically: COLMAP uses Y-down but video expects Y-up
        if flip:
            frame = np.flip(frame, axis=0)
        writer.append_data(frame)
    
    writer.close()
    print(f"Dual trajectory video saved to {output_path}")


def save_trajectory(
    camtoworlds: np.ndarray,
    output_path: Path,
    intrinsics: Optional[np.ndarray] = None,
    image_size: Optional[tuple] = None,
) -> None:
    """
    Save trajectory to file.
    """
    output_path = Path(output_path)
    
    if output_path.suffix == ".npz":
        data = {"camtoworlds": camtoworlds}
        if intrinsics is not None:
            data["intrinsics"] = intrinsics
        if image_size is not None:
            data["image_size"] = np.array(image_size)
        np.savez(output_path, **data)
    else:
        np.save(output_path, camtoworlds)
    
    print(f"Saved trajectory with {len(camtoworlds)} poses to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and render camera trajectories for Gaussian Splatting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render interpolated trajectory with camera moved up 10cm
  python generate_traj.py --data-dir ./data --ckpt ./ckpts/ckpt.pt -o video.mp4 --up-offset 0.1

  # Render ellipse trajectory  
  python generate_traj.py --data-dir ./data --ckpt ./ckpts/ckpt.pt -o video.mp4 --traj-type ellipse

  # Render using original COLMAP camera poses (from images.bin)
  python generate_traj.py --data-dir ./data --ckpt ./ckpts/ckpt.pt -o video.mp4 --traj-type colmap

  # Render using COLMAP poses with 3x interpolation for smoother video
  python generate_traj.py --data-dir ./data --ckpt ./ckpts/ckpt.pt -o video.mp4 --traj-type colmap --colmap-interp 3

  # Save trajectory without rendering (no checkpoint needed)
  python generate_traj.py --data-dir ./data -o traj.npz --save-traj-only
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Path to COLMAP dataset directory"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint file (required for video rendering)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str, 
        required=False,
        help="Output file (.mp4 for video, .npy/.npz for trajectory)"
    )
    parser.add_argument(
        "--traj-type", 
        type=str,
        choices=["interp", "ellipse", "spiral", "colmap", "wall"],
        default="interp",
        help="Trajectory type (default: interp). 'colmap' uses original COLMAP camera poses."
    )
    parser.add_argument(
        "--colmap-interp",
        type=int,
        default=1,
        help="Interpolation factor for colmap trajectory (1 = use original poses, >1 = interpolate)"
    )
    parser.add_argument(
        "--traj-file",
        type=str,
        default=None,
        help="Load trajectory from file (.npy/.npz) instead of generating. Overrides --traj-type."
    )
    parser.add_argument(
        "--offset-x", 
        type=float, 
        default=0.0,
        help="X offset in meters (world space)"
    )
    parser.add_argument(
        "--offset-y", 
        type=float, 
        default=0.0,
        help="Y offset in meters (world space)"
    )
    parser.add_argument(
        "--offset-z", 
        type=float, 
        default=0.0,
        help="Z offset in meters (up is -Z, so use negative to go up)"
    )
    parser.add_argument(
        "--up-offset",
        type=float,
        default=None,
        help="Shorthand for moving camera up (in meters). Equivalent to --offset-z with sign flipped."
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Number of frames in trajectory (None uses defaults)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (limits trajectory length after generation/loading)"
    )
    parser.add_argument(
        "--data-factor", 
        type=int, 
        default=1,
        help="Downsample factor for images (default: 1 for full resolution)"
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=None,
        help="Scene scale (overrides auto-computed value)"
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Spherical harmonics degree (default: 3)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second (default: 30)"
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Don't show depth alongside RGB in video"
    )
    parser.add_argument(
        "--save-traj-only",
        action="store_true",
        help="Only save trajectory file, don't render video"
    )
    parser.add_argument(
        "--near-plane",
        type=float,
        default=0.01,
        help="Near clipping plane (default: 0.01)"
    )
    parser.add_argument(
        "--far-plane",
        type=float,
        default=1e10,
        help="Far clipping plane (default: 1e10)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip rendering trajectory comparison visualization"
    )
    parser.add_argument(
        "--dual-video",
        action="store_true",
        help="Render video with two rows: original trajectory on top, offset on bottom"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip images vertically (use if images appear upside down)"
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Pitch of viewing"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="elevateion of viewing"
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine if we're rendering video
    output_path = Path(args.output)
    is_video = output_path.suffix.lower() in [".mp4", ".avi", ".mov", ".webm"]
    
    if is_video and args.ckpt is None and not args.save_traj_only:
        parser.error("--ckpt is required for video rendering")
    
    # Load parser
    print(f"Loading COLMAP data from {args.data_dir}...")
    
    # Set up image directory if needed (for trajectory generation, images aren't strictly required)
    try:
        setup_image_directory(Path(args.data_dir), factor=args.data_factor)
    except Exception as e:
        print(f"Warning: Could not set up image directory: {e}")
        print("  Continuing anyway - images may not be needed for trajectory generation")
    
    colmap_parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=8,
    )
    
    # Determine scene scale
    if args.scene_scale is not None:
        _scene_scale = args.scene_scale
    else:
        _scene_scale = colmap_parser.scene_scale * 1.1
    print(f"Using scene_scale={_scene_scale:.4f}")
    
    # Handle up_offset shorthand
    offset_z = args.offset_z
    if args.up_offset is not None:
        offset_z = -args.up_offset  # up is -Z
    
    # Load or generate trajectory
    if args.traj_file is not None:
        # Load trajectory from file
        traj_path = Path(args.traj_file)
        if not traj_path.exists():
            print(f"ERROR: Trajectory file not found: {traj_path}")
            print(f"  Current directory: {Path.cwd()}")
            print(f"  Try using absolute path or check file location.")
            print(f"  Example: --traj-file {Path.cwd() / 'my_traj.npy'}")
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        print(f"Loading trajectory from {traj_path}...")

        if traj_path.suffix.lower() == ".json":
            # Load from JSON (produced by generate_trajectory.py)
            # Apply parser transform for normalization
            original_camtoworlds = load_trajectory_from_json(
                traj_path,
                parser_transform=colmap_parser.transform,
            )
        else:
            # Load from npy/npz
            original_camtoworlds = load_trajectory(traj_path, convert_zup=False)
        print(f"Loaded {len(original_camtoworlds)} poses from file")
    else:
        # Generate trajectory from parser
        print(f"Generating {args.traj_type} trajectory...")
        original_camtoworlds = generate_trajectory_from_parser(
            colmap_parser,
            traj_type=args.traj_type,
            n_frames=args.n_frames,
            scene_scale=_scene_scale,
            colmap_interp=args.colmap_interp,
            elevation=args.elevation,
            pitch_degrees=args.pitch,
        )
    
    # Apply max_frames limit if specified
    if args.max_frames is not None and args.max_frames > 0:
        if len(original_camtoworlds) > args.max_frames:
            print(f"Limiting trajectory from {len(original_camtoworlds)} to {args.max_frames} frames")
            original_camtoworlds = original_camtoworlds[:args.max_frames]
    
    # Apply offset if any
    offset = np.array([args.offset_x, args.offset_y, offset_z])
    has_offset = np.any(offset != 0)
    
    if has_offset:
        # Scale offset by scene_scale to convert from meters to scene units
        scaled_offset = offset / _scene_scale
        print(f"Applying offset: world={offset} -> scene_units={scaled_offset}")
        offset_camtoworlds = offset_trajectory(original_camtoworlds, scaled_offset)
    else:
        offset_camtoworlds = original_camtoworlds
    
    # Apply max_frames limit to offset trajectory as well
    if args.max_frames is not None and args.max_frames > 0:
        if len(offset_camtoworlds) > args.max_frames:
            offset_camtoworlds = offset_camtoworlds[:args.max_frames]
    
    print(f"Using {len(offset_camtoworlds)} poses")
    
    # Get camera intrinsics and image size
    first_camera_id = list(colmap_parser.Ks_dict.keys())[0]
    K = colmap_parser.Ks_dict[first_camera_id]
    width, height = colmap_parser.imsize_dict[first_camera_id]
    
    if args.save_traj_only or not is_video:
        # Just save trajectory
        save_trajectory(offset_camtoworlds, output_path, intrinsics=K, image_size=(width, height))
    else:
        # Load checkpoint and render video
        print(f"Loading checkpoint from {args.ckpt}...")
        splats, step = load_checkpoint(args.ckpt, device=device)
        print(f"Loaded checkpoint from step {step} with {len(splats['means'])} Gaussians")
        
        # Render trajectory comparison if there's an offset
        if has_offset and not args.no_comparison:
            # Create comparison output paths
            comparison_image_path = str(output_path.with_stem(output_path.stem + "_comparison").with_suffix(".png"))
            comparison_video_path = str(output_path.with_stem(output_path.stem + "_comparison"))
            
            # Use a viewpoint from the middle of the original trajectory for static comparison
            mid_idx = len(original_camtoworlds) // 2
            viewpoint = original_camtoworlds[mid_idx]
            
            # Render static comparison image
            print("Rendering trajectory comparison image...")
            render_trajectory_comparison(
                splats=splats,
                original_trajectory=original_camtoworlds,
                offset_trajectory_poses=offset_camtoworlds,
                viewpoint_camtoworld=viewpoint,
                K=K,
                width=width,
                height=height,
                output_path=comparison_image_path,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                sh_degree=args.sh_degree,
                device=device,
                flip=args.flip,
            )
            
            # Render comparison video showing both trajectories from the original trajectory viewpoints
            # Use a subset of frames for the comparison video
            comparison_frames = original_camtoworlds[::max(1, len(original_camtoworlds) // 120)]
            print("Rendering trajectory comparison video...")
            render_trajectory_comparison_video(
                splats=splats,
                original_trajectory=original_camtoworlds,
                offset_trajectory_poses=offset_camtoworlds,
                viewpoint_trajectory=comparison_frames,
                K=K,
                width=width,
                height=height,
                output_path=comparison_video_path,
                fps=args.fps,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                sh_degree=args.sh_degree,
                device=device,
                flip=args.flip,
            )
        
        # Render video
        if args.dual_video and has_offset:
            # Render dual trajectory video (original on top, offset on bottom)
            print("Rendering dual trajectory video (original + offset)...")
            render_dual_trajectory_video(
                splats=splats,
                original_camtoworlds=original_camtoworlds,
                offset_camtoworlds=offset_camtoworlds,
                K=K,
                width=width,
                height=height,
                output_path=str(output_path),
                fps=args.fps,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                sh_degree=args.sh_degree,
                device=device,
                flip=args.flip,
            )
        else:
            # Render main video from offset trajectory
            print("Rendering main trajectory video...")
            render_trajectory_to_video(
                splats=splats,
                camtoworlds=offset_camtoworlds,
                K=K,
                width=width,
                height=height,
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
