"""Trajectory utilities for camera path generation and manipulation."""

from pathlib import Path

import click
import numpy as np

from colmap_loader import load_pcd_from_dir, setup_image_directory
from datasets.colmap import Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from trajectory_data import TrajectoryData, TrajectoryType, WallTrajectoryShape
from viz_utils import view_pcd_with_traj


# ── Low-level helpers ────────────────────────────────────────────────────────


def ensure_4x4(camtoworlds: np.ndarray) -> np.ndarray:
    """Ensure [N, 3, 4] or [N, 4, 4] camera matrices are [N, 4, 4]."""
    if camtoworlds.shape[1] == 3:
        n = len(camtoworlds)
        bottom = np.tile(np.array([[[0.0, 0.0, 0.0, 1.0]]]), (n, 1, 1))
        camtoworlds = np.concatenate([camtoworlds, bottom], axis=1)
    return camtoworlds


def _build_camera_pose(
    pos: np.ndarray,
    forward_xy: np.ndarray,
    pitch_rad: float,
) -> np.ndarray:
    """
    Build a 4x4 camera-to-world matrix (Z-up world, COLMAP/OpenCV camera convention).

    In COLMAP convention the camera X axis is right, Y is down, Z is forward.

    Args:
        pos: Camera world position [3,].
        forward_xy: Horizontal look direction in the XY plane [2,] (unnormalised).
        pitch_rad: Pitch angle in radians (positive = tilt down).

    Returns:
        4x4 camera-to-world matrix.
    """
    fxy = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)
    forward = np.array([fxy[0], fxy[1], 0.0])

    # Right is perpendicular to forward in the XY plane (Z-up world).
    right = np.array([-forward[1], forward[0], 0.0])
    right /= np.linalg.norm(right) + 1e-8

    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    world_up = np.array([0.0, 0.0, 1.0])

    # Rodrigues rotation of forward and up around the right axis.
    forward_pitched = forward * cos_p + np.cross(right, forward) * sin_p
    up_pitched = world_up * cos_p + np.cross(right, world_up) * sin_p

    forward_pitched /= np.linalg.norm(forward_pitched) + 1e-8
    up_pitched /= np.linalg.norm(up_pitched) + 1e-8

    # Recompute right to ensure orthogonality.
    right = np.cross(up_pitched, forward_pitched)
    right /= np.linalg.norm(right) + 1e-8

    # COLMAP/OpenCV: X=right, Y=-up (down), Z=forward.
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, -up_pitched, forward_pitched])
    c2w[:3, 3] = pos
    return c2w


# ── Trajectory generators ────────────────────────────────────────────────────


def generate_wall_trajectory(
    camtoworlds: np.ndarray,
    n_frames: int = 120,
    height: float | None = None,
    pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    shape: WallTrajectoryShape = WallTrajectoryShape.ELLIPSE,
) -> np.ndarray:
    """
    Generate a trajectory around the room perimeter, facing inward and slightly down.

    Args:
        camtoworlds: Reference camera poses [N, 3, 4] or [N, 4, 4].
        n_frames: Number of frames in the output trajectory.
        height: Camera Z height. None uses the mean height of reference cameras.
        pitch_degrees: Downward pitch angle in degrees (positive = looking down).
        wall_distance: Distance multiplier (1.0 = at camera bounding box).
        shape: Path shape ("ellipse" or "rectangle").

    Returns:
        Camera-to-world transforms [N, 4, 4].

    Note:
        Assumes Z-up world convention.
    """
    positions = camtoworlds[:, :3, 3]
    cx = (positions[:, 0].max() + positions[:, 0].min()) / 2
    cy = (positions[:, 1].max() + positions[:, 1].min()) / 2
    ex = (positions[:, 0].max() - positions[:, 0].min()) / 2
    ey = (positions[:, 1].max() - positions[:, 1].min()) / 2

    if height is None:
        height = float(positions[:, 2].mean())

    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    if shape == "ellipse":
        x = cx + wall_distance * ex * np.cos(t)
        y = cy + wall_distance * ey * np.sin(t)
    else:
        x, y = np.zeros(n_frames), np.zeros(n_frames)
        for i, ti in enumerate(t):
            s = (ti / (2 * np.pi)) * 4
            if s < 1:
                x[i] = cx + wall_distance * ex
                y[i] = cy - wall_distance * ey + 2 * wall_distance * ey * s
            elif s < 2:
                x[i] = cx + wall_distance * ex - 2 * wall_distance * ex * (s - 1)
                y[i] = cy + wall_distance * ey
            elif s < 3:
                x[i] = cx - wall_distance * ex
                y[i] = cy + wall_distance * ey - 2 * wall_distance * ey * (s - 2)
            else:
                x[i] = cx - wall_distance * ex + 2 * wall_distance * ex * (s - 3)
                y[i] = cy - wall_distance * ey

    pitch_rad = np.radians(pitch_degrees)
    center = np.array([cx, cy])
    result = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        result[i] = _build_camera_pose(
            pos=np.array([x[i], y[i], height]),
            forward_xy=center - np.array([x[i], y[i]]),
            pitch_rad=pitch_rad,
        )
    return result


def generate_elevated_trajectory(
    camtoworlds: np.ndarray,
    height: float | None = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 1,
    subsample: int = 1,
    look_at_center: bool = True,
) -> np.ndarray:
    """
    Follow original camera XY positions at a fixed Z height with a given pitch.

    Args:
        camtoworlds: Reference camera poses [N, 3, 4] or [N, 4, 4].
        height: Camera Z height. None uses the mean height of reference cameras.
        pitch_degrees: Downward pitch angle in degrees.
        n_interp: Interpolation points between each original camera.
        subsample: Use every Nth original camera position.
        look_at_center: If True cameras face scene center; else follow path direction.

    Returns:
        Camera-to-world transforms [N, 4, 4].

    Note:
        Assumes Z-up world convention.
    """
    positions = camtoworlds[::subsample, :3, 3]
    cx = (positions[:, 0].max() + positions[:, 0].min()) / 2
    cy = (positions[:, 1].max() + positions[:, 1].min()) / 2
    center = np.array([cx, cy])

    if height is None:
        height = float(positions[:, 2].mean())

    elevated = positions.copy()
    elevated[:, 2] = height

    if n_interp > 1:
        from scipy.interpolate import interp1d

        n = len(elevated)
        t_orig = np.arange(n)
        t_new = np.linspace(0, n - 1, n + (n - 1) * (n_interp - 1))
        elevated = interp1d(t_orig, elevated, axis=0, kind="cubic")(t_new)

    n_frames = len(elevated)
    pitch_rad = np.radians(pitch_degrees)
    result = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        pos = elevated[i]
        if look_at_center:
            fwd_xy = center - pos[:2]
        elif i < n_frames - 1:
            fwd_xy = elevated[i + 1, :2] - pos[:2]
        else:
            fwd_xy = pos[:2] - elevated[i - 1, :2]
        result[i] = _build_camera_pose(pos, fwd_xy, pitch_rad)
    return result


def generate_trajectory_from_selected_points(
    selected_positions: np.ndarray,
    height: float | None = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 5,
    look_at_center: bool = True,
    closed_loop: bool = False,
    center_override: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate a smooth trajectory from user-selected 2D or 3D waypoints.

    Args:
        selected_positions: Selected positions [N, 2] (XY) or [N, 3] (XYZ).
        height: Override Z for all positions. None uses existing Z (or 0 for 2D input).
        pitch_degrees: Downward pitch angle in degrees.
        n_interp: Interpolation points between consecutive keypoints.
        look_at_center: If True cameras face the centroid; else follow path direction.
        closed_loop: If True connect the last point back to the first.
        center_override: If provided, use this XY point [2,] as the look-at center.

    Returns:
        Camera-to-world transforms [N, 4, 4].

    Note:
        Assumes Z-up world convention.
    """
    positions = np.array(selected_positions, dtype=float)
    if positions.shape[1] == 2:
        if height is None:
            height = 0.0
        positions = np.column_stack([positions, np.full(len(positions), height)])
    elif height is not None:
        positions[:, 2] = height

    center: np.ndarray = (
        np.array(center_override[:2])
        if center_override is not None
        else positions[:, :2].mean(axis=0)
    )

    if closed_loop:
        positions = np.vstack([positions, positions[:1]])

    if n_interp > 1:
        from scipy.interpolate import CubicSpline, interp1d

        if closed_loop and len(positions) >= 3:
            # Sort by angle around centre for a consistent winding order.
            rel = positions[:, :2] - center
            positions = positions[np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))]
            pos_per = np.vstack([positions, positions[:1]])
            t_orig = np.linspace(0.0, 1.0, len(pos_per))
            n_total = len(positions) + (len(positions) - 1) * (n_interp - 1)
            t_new = np.linspace(0.0, 1.0, n_total)
            spline = CubicSpline(t_orig, pos_per, axis=0, bc_type="periodic")
            positions = np.vstack([spline(t_new), spline(t_new[:1])])
        else:
            n_key = len(positions)
            t_orig = np.linspace(0.0, 1.0, n_key)
            n_total = n_key + (n_key - 1) * (n_interp - 1)
            t_new = np.linspace(0.0, 1.0, n_total)
            positions = interp1d(t_orig, positions, axis=0, kind="cubic")(t_new)

    n_frames = len(positions)
    pitch_rad = np.radians(pitch_degrees)
    result = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        pos = positions[i]
        if look_at_center:
            fwd_xy = center - pos[:2]
        elif i < n_frames - 1:
            fwd_xy = positions[i + 1, :2] - pos[:2]
        else:
            fwd_xy = pos[:2] - positions[i - 1, :2]
        result[i] = _build_camera_pose(pos, fwd_xy, pitch_rad)
    return result


# ── Coordinate-system transforms ─────────────────────────────────────────────


def convert_zup_to_colmap(camtoworlds: np.ndarray) -> np.ndarray:
    """
    Convert camera poses from Z-up world convention to COLMAP/OpenCV convention.

    COLMAP: X right, Y down, Z forward.
    Z-up:   X right, Y forward, Z up.

    Args:
        camtoworlds: Camera-to-world transforms [N, 4, 4] in Z-up convention.

    Returns:
        Camera-to-world transforms [N, 4, 4] in COLMAP convention.
    """
    T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    result = camtoworlds.copy()
    result[:, :3, 3] = (T @ camtoworlds[:, :3, 3].T).T
    result[:, :3, :3] = T @ camtoworlds[:, :3, :3] @ T.T
    return result


def offset_trajectory(camtoworlds: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Apply a world-space translation to all camera positions.

    Args:
        camtoworlds: Camera-to-world transforms [N, 3, 4] or [N, 4, 4].
        offset: World-space offset vector [3,].

    Returns:
        Modified camera transforms with the offset applied to positions.
    """
    result = camtoworlds.copy()
    result[..., :3, 3] += offset
    return result


def transform_trajectory(
    camtoworlds: np.ndarray,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply scale, rotation, and translation to a camera trajectory.

    Args:
        camtoworlds: Camera-to-world transforms [N, 4, 4].
        rotation: Optional 3x3 rotation matrix applied in world space.
        translation: Optional 3D translation applied in world space.
        scale: Scale factor applied to camera positions.

    Returns:
        Transformed camera-to-world matrices [N, 4, 4].
    """
    result = camtoworlds.copy()
    if scale != 1.0:
        result[..., :3, 3] *= scale
    if rotation is not None:
        result[:, :3, 3] = (rotation @ result[:, :3, 3].T).T
        result[:, :3, :3] = rotation @ result[:, :3, :3]
    if translation is not None:
        result[..., :3, 3] += translation
    return result


def apply_elevation_and_pitch(
    camtoworlds: np.ndarray,
    elevation: float | None = None,
    pitch_degrees: float = 0.0,
) -> np.ndarray:
    """
    Translate cameras in Z (elevation) and rotate each around its own right axis (pitch).

    Args:
        camtoworlds: Camera-to-world transforms [N, 3, 4] or [N, 4, 4].
        elevation: Z translation (positive = up). None or 0 means no translation.
        pitch_degrees: Pitch angle in degrees to tilt around the camera right axis
                       (positive = look down).

    Returns:
        Transformed camera-to-world matrices [N, 4, 4].
    """
    result = ensure_4x4(camtoworlds.copy())
    if elevation:
        result[..., :3, 3] += np.array([0.0, 0.0, elevation])
    if pitch_degrees != 0.0:
        pitch_rad = np.radians(pitch_degrees)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        for i in range(len(result)):
            R = result[i, :3, :3]
            ax = R[:, 0]  # Camera right axis in world space.
            K_sk = np.array(
                [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]
            )
            Rp = np.eye(3) + sin_p * K_sk + (1 - cos_p) * (K_sk @ K_sk)
            result[i, :3, :3] = Rp @ R
    return result


# ── High-level trajectory builder ────────────────────────────────────────────


def generate_trajectory_from_parser(
    parser: Parser,
    traj_type: TrajectoryType = TrajectoryType.INTERP,
    n_frames: int | None = None,
    scene_scale: float = 1.0,
    elevation: float | None = None,
    pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: WallTrajectoryShape = WallTrajectoryShape.ELLIPSE,
    elevated_subsample: int = 1,
    elevated_look_at_center: bool = True,
    colmap_interp: int = 1,
) -> np.ndarray:
    """
    Generate a camera trajectory from a COLMAP Parser.

    Args:
        parser: COLMAP Parser with camera poses.
        traj_type: Trajectory type.
        n_frames: Desired number of output frames.
        scene_scale: Scene scale factor (used for spiral trajectory).
        elevation: Camera Z height for wall/elevated trajectories. None uses mean.
        pitch_degrees: Downward pitch angle in degrees.
        wall_distance: Wall distance multiplier for the wall trajectory.
        wall_shape: Path shape for the wall trajectory.
        elevated_subsample: Use every Nth camera for elevated trajectory.
        elevated_look_at_center: Elevated cameras face scene centre if True.
        colmap_interp: Interpolation factor for the colmap trajectory.

    Returns:
        Camera-to-world transforms [N, 4, 4].
    """
    poses = parser.camtoworlds[5:-5]

    if traj_type == "interp":
        n_interp = 1 if n_frames is None else max(1, n_frames // len(poses))
        poses = generate_interpolated_path(poses, n_interp)
        poses = apply_elevation_and_pitch(poses, elevation, pitch_degrees)
    elif traj_type == "ellipse":
        n = 120 if n_frames is None else n_frames
        poses = generate_ellipse_path_z(poses, n_frames=n, height=poses[:, 2, 3].mean())
    elif traj_type == "spiral":
        n = 120 if n_frames is None else n_frames
        poses = generate_spiral_path(
            poses,
            bounds=parser.bounds * scene_scale,
            n_frames=n,
            spiral_scale_r=parser.extconf.get("spiral_radius_scale", 1.0),
        )
    elif traj_type == "wall":
        n = 120 if n_frames is None else n_frames
        poses = generate_wall_trajectory(
            poses,
            n_frames=n,
            height=elevation,
            pitch_degrees=pitch_degrees,
            wall_distance=wall_distance,
            shape=wall_shape,
        )
    elif traj_type == "elevated":
        n_interp = (
            1
            if n_frames is None
            else max(1, n_frames // (len(poses) // elevated_subsample))
        )
        poses = generate_elevated_trajectory(
            poses,
            height=elevation,
            pitch_degrees=pitch_degrees,
            n_interp=n_interp,
            subsample=elevated_subsample,
            look_at_center=elevated_look_at_center,
        )
    elif traj_type == "colmap":
        poses = parser.camtoworlds.copy()
        if colmap_interp > 1:
            poses = generate_interpolated_path(poses, colmap_interp)
        print(f"Using original COLMAP camera poses: {len(poses)} poses")
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return ensure_4x4(poses)


# ── Persistence helpers ───────────────────────────────────────────────────────


def load_trajectory(path: Path, convert_zup: bool = True) -> np.ndarray:
    """
    Load a trajectory from a .npy or .npz file.

    Args:
        path: Path to trajectory file.
        convert_zup: If True, convert from Z-up to COLMAP convention.

    Returns:
        Camera-to-world transforms [N, 4, 4].
    """
    path = Path(path)
    camtoworlds = (
        np.load(path)["camtoworlds"] if path.suffix == ".npz" else np.load(path)
    )
    return convert_zup_to_colmap(camtoworlds) if convert_zup else camtoworlds


def save_baseline_trajectory(
    data_dir: Path,
    output_path: Path | None = None,
    transform_path: Path | None = None,
    factor: int = 1,
) -> Path:
    """
    Save baseline.npy with original COLMAP camera poses.

    Args:
        data_dir: Path to COLMAP dataset directory.
        output_path: Destination for baseline.npy. Defaults to data_dir/baseline.npy.
        transform_path: Destination for transform.npy. Defaults to data_dir/transform.npy.
        factor: Downsample factor for images.

    Returns:
        Path to the saved baseline.npy file.
    """
    setup_image_directory(data_dir, factor)
    parser = Parser(data_dir=data_dir.as_posix(), factor=factor, normalize=True, test_every=8)

    output_path = Path(output_path) if output_path is not None else data_dir / "baseline.npy"
    transform_path = (
        Path(transform_path) if transform_path is not None else data_dir / "transform.npy"
    )

    np.save(output_path, parser.camtoworlds)
    np.save(transform_path, parser.transform)
    print(f"Saved {len(parser.camtoworlds)} poses to {output_path}")
    return output_path


# ── Fixed trajectory generation ───────────────────────────────────────────────


def _parser_intrinsics(
    parser: Parser,
) -> tuple[np.ndarray, int, int]:
    """Extract intrinsics and image size from the first camera in a Parser."""
    first_cam = list(parser.Ks_dict.keys())[0]
    K = parser.Ks_dict[first_cam]
    img_w, img_h = parser.imsize_dict[first_cam]
    return K, img_w, img_h


def generate_fixed_trajectory(
    data_dir: Path,
    traj_type: TrajectoryType = TrajectoryType.INTERP,
    n_frames: int | None = None,
    elevation: float | None = None,
    pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: WallTrajectoryShape = WallTrajectoryShape.ELLIPSE,
    elevated_subsample: int = 1,
    elevated_look_at_center: bool = True,
    colmap_interp: int = 1,
    output_path: Path | None = None,
    visualize: bool = True,
) -> TrajectoryData:
    """
    Generate a fixed (non-interactive) trajectory and return a TrajectoryData.

    Args:
        data_dir: Path to the COLMAP dataset directory.
        traj_type: Trajectory type.
        n_frames: Desired number of output frames.
        elevation: Camera Z height. None uses the parser mean.
        pitch_degrees: Downward pitch angle in degrees.
        wall_distance: Wall distance multiplier for the wall trajectory.
        wall_shape: Path shape for the wall trajectory.
        elevated_subsample: Use every Nth camera for elevated trajectory.
        elevated_look_at_center: Elevated cameras face scene centre if True.
        colmap_interp: Interpolation factor for the colmap trajectory.
        output_path: If provided, save the trajectory to this .npz file.
        visualize: If True open an Open3D window showing the trajectory.

    Returns:
        TrajectoryData instance with the generated trajectory.
    """
    colmap = load_pcd_from_dir(data_dir, save=False)
    parser = Parser(data_dir=str(data_dir), factor=1, normalize=True, test_every=8)
    scene_scale = parser.scene_scale * 1.1

    traj = generate_trajectory_from_parser(
        parser,
        traj_type=traj_type,
        n_frames=n_frames,
        scene_scale=scene_scale,
        elevation=elevation,
        pitch_degrees=pitch_degrees,
        wall_distance=wall_distance,
        wall_shape=wall_shape,
        elevated_subsample=elevated_subsample,
        elevated_look_at_center=elevated_look_at_center,
        colmap_interp=colmap_interp,
    )

    K, img_w, img_h = _parser_intrinsics(parser)
    traj_data = TrajectoryData(
        camtoworlds=traj,
        intrinsics=K,
        width=img_w,
        height=img_h,
        metadata={
            "traj_type": traj_type,
            "n_frames": len(traj),
            "elevation": float(elevation) if elevation is not None else None,
            "pitch_degrees": float(pitch_degrees),
        },
    )

    if output_path is not None:
        traj_data.save(output_path)

    if visualize:
        print("\nShowing generated trajectory...")
        view_pcd_with_traj(colmap, [traj_data])

    return traj_data


# ── Interactive picking ───────────────────────────────────────────────────────


def interactive_pick_trajectory(
    data_dir: Path,
    height: float | None = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 5,
    look_at_center: bool = True,
    closed_loop: bool = False,
    pick_center: bool = False,
    output_path: Path | None = None,
    visualize: bool = True,
) -> TrajectoryData | None:
    """
    Interactively pick waypoints (and optionally a focus point) to build a trajectory.

    Opens one or two Open3D picking windows:
    1. (Optional) Shift+Click to pick a centre/focus point. Last click is used.
    2. Shift+Click to pick waypoints in traversal order. Press Q when done.

    Args:
        data_dir: Path to the COLMAP dataset directory.
        height: Camera Z height for all waypoints. None auto-selects from scene.
        pitch_degrees: Downward pitch angle in degrees.
        n_interp: Interpolation points between consecutive keypoints.
        look_at_center: If True cameras face the focus/centroid; else follow path.
        closed_loop: If True connect the last waypoint back to the first.
        pick_center: If True show a first window to pick the look-at centre.
        output_path: If provided, save the trajectory to this .npz file.
        visualize: If True open an Open3D window showing the result.

    Returns:
        TrajectoryData instance, or None if fewer than 2 waypoints were selected.
    """
    import open3d as o3d

    print(f"Loading COLMAP data from {data_dir}...")
    colmap = load_pcd_from_dir(data_dir, save=False)
    pcd = colmap.viz_o3d(show_cam=True)

    if height is None:
        height = float(colmap.points[:, 2].mean())
        print(f"Auto height: {height:.3f}")

    combined_points = np.vstack([colmap.points, colmap.cam_positions])

    center_point_xy: np.ndarray | None = None

    # ── Step 1 (optional): pick focus/centre ─────────────────────────────────
    if pick_center and look_at_center:
        print("\n=== STEP 1: PICK CENTER POINT (Shift+Click, Q when done) ===")
        vc = o3d.visualization.VisualizerWithEditing()
        vc.create_window(window_name="Pick Center Point", width=1280, height=720)
        vc.add_geometry(pcd)
        vc.run()
        picked = vc.get_picked_points()
        vc.destroy_window()

        if picked:
            center_pos = combined_points[picked[-1]]
            center_point_xy = center_pos[:2]
            print(f"Center: XY=({center_pos[0]:.3f}, {center_pos[1]:.3f})")
        else:
            print("No centre selected; will use waypoint centroid.")

    # ── Step 2: pick waypoints ────────────────────────────────────────────────
    step = 2 if (pick_center and look_at_center) else 1
    print(f"\n=== STEP {step}: PICK WAYPOINTS (Shift+Click in order, Q when done) ===")
    print(f"  Height: {height:.3f}  Pitch: {pitch_degrees}°  Closed loop: {closed_loop}")
    vw = o3d.visualization.VisualizerWithEditing()
    vw.create_window(window_name="Pick Waypoints", width=1280, height=720)
    vw.add_geometry(pcd)
    vw.run()
    picked_wp = vw.get_picked_points()
    vw.destroy_window()

    if len(picked_wp) < 2:
        print(f"ERROR: Need ≥ 2 waypoints, got {len(picked_wp)}.")
        return None

    waypoints_xy = np.array([combined_points[idx][:2] for idx in picked_wp])
    waypoints_3d = np.column_stack([waypoints_xy, np.full(len(waypoints_xy), height)])

    center_for_lookat: np.ndarray | None = None
    if look_at_center:
        center_for_lookat = (
            center_point_xy if center_point_xy is not None else waypoints_xy.mean(axis=0)
        )

    trajectory = generate_trajectory_from_selected_points(
        waypoints_3d,
        height=height,
        pitch_degrees=pitch_degrees,
        n_interp=n_interp,
        look_at_center=look_at_center,
        closed_loop=closed_loop,
        center_override=center_for_lookat,
    )

    parser = Parser(data_dir=str(data_dir), factor=1, normalize=True, test_every=8)
    K, img_w, img_h = _parser_intrinsics(parser)

    traj_data = TrajectoryData(
        camtoworlds=trajectory,
        intrinsics=K,
        width=img_w,
        height=img_h,
        waypoints=waypoints_xy,
        center_point=center_point_xy,
        metadata={
            "height": float(height),
            "pitch_degrees": float(pitch_degrees),
            "n_interp": int(n_interp),
            "look_at_center": bool(look_at_center),
            "closed_loop": bool(closed_loop),
            "pick_center": bool(pick_center),
        },
    )

    if output_path is not None:
        traj_data.save(output_path)

    if visualize:
        print("\nShowing generated trajectory...")
        view_pcd_with_traj(colmap, [traj_data])

    return traj_data


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """Generate and inspect camera trajectories from COLMAP data."""


@cli.command("fixed")
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-t",
    "--traj-type",
    type=click.Choice(TrajectoryType),
    default=TrajectoryType.INTERP,
    show_default=True,
    help="Trajectory type.",
)
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None,
    help="Save trajectory to .npz file.",
)
@click.option("--n-frames", type=int, default=None, help="Desired number of frames.")
@click.option("--elevation", type=float, default=None, help="Camera Z height.")
@click.option(
    "--pitch", type=float, default=15.0, show_default=True,
    help="Downward pitch in degrees.",
)
@click.option("--wall-distance", type=float, default=1.2, show_default=True)
@click.option(
    "--wall-shape",
    type=click.Choice(WallTrajectoryShape),
    default=WallTrajectoryShape.ELLIPSE,
    show_default=True,
)
@click.option("--elevated-subsample", type=int, default=1, show_default=True)
@click.option(
    "--no-look-at-center", is_flag=True, default=False,
    help="Elevated cameras follow path direction instead of facing the centre.",
)
@click.option("--colmap-interp", type=int, default=1, show_default=True)
@click.option(
    "--no-viz", is_flag=True, default=False, help="Skip the Open3D visualisation."
)
def cmd_fixed(
    data_dir: Path,
    traj_type: TrajectoryType,
    output: Path | None,
    n_frames: int | None,
    elevation: float | None,
    pitch: float,
    wall_distance: float,
    wall_shape: WallTrajectoryShape,
    elevated_subsample: int,
    no_look_at_center: bool,
    colmap_interp: int,
    no_viz: bool,
) -> None:
    """Generate a fixed camera trajectory (no interactive picking)."""
    generate_fixed_trajectory(
        data_dir=data_dir,
        traj_type=traj_type,
        n_frames=n_frames,
        elevation=elevation,
        pitch_degrees=pitch,
        wall_distance=wall_distance,
        wall_shape=wall_shape,
        elevated_subsample=elevated_subsample,
        elevated_look_at_center=not no_look_at_center,
        colmap_interp=colmap_interp,
        output_path=output,
        visualize=not no_viz,
    )


@cli.command("pick")
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None,
    help="Save trajectory to .npz file.",
)
@click.option("--height", type=float, default=None, help="Camera Z height (None = auto).")
@click.option(
    "--pitch", type=float, default=15.0, show_default=True,
    help="Downward pitch in degrees.",
)
@click.option("--n-interp", type=int, default=5, show_default=True)
@click.option("--closed-loop", is_flag=True, default=False)
@click.option(
    "--look-along-path", is_flag=True, default=False,
    help="Cameras follow path direction instead of facing the centre.",
)
@click.option(
    "--pick-center", is_flag=True, default=False,
    help="First pick a look-at focus point, then pick waypoints.",
)
@click.option(
    "--no-viz", is_flag=True, default=False, help="Skip the Open3D visualisation."
)
def cmd_pick(
    data_dir: Path,
    output: Path | None,
    height: float | None,
    pitch: float,
    n_interp: int,
    closed_loop: bool,
    look_along_path: bool,
    pick_center: bool,
    no_viz: bool,
) -> None:
    """Interactively pick waypoints to build a camera trajectory."""
    interactive_pick_trajectory(
        data_dir=data_dir,
        height=height,
        pitch_degrees=pitch,
        n_interp=n_interp,
        look_at_center=not look_along_path,
        closed_loop=closed_loop,
        pick_center=pick_center,
        output_path=output,
        visualize=not no_viz,
    )


@cli.command("baseline")
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None,
    help="Destination for baseline.npy (default: data_dir/baseline.npy).",
)
def cmd_baseline(data_dir: Path, output: Path | None) -> None:
    """Save baseline.npy with original COLMAP camera poses to data_dir."""
    save_baseline_trajectory(data_dir, output_path=output)


if __name__ == "__main__":
    cli()
