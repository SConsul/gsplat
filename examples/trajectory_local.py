"""
Trajectory utilities for camera path generation and manipulation.
"""

from pathlib import Path
from typing import Optional, Literal, Dict, Tuple, List
import numpy as np

from datasets.colmap import Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)


def generate_wall_trajectory(
    camtoworlds: np.ndarray,
    n_frames: int = 120,
    height: Optional[float] = None,
    pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    shape: Literal["rectangle", "ellipse"] = "ellipse",
) -> np.ndarray:
    """
    Generate a trajectory around the room perimeter, facing inward and slightly down.
    
    This creates a path near the "walls" of the scene (the outer boundary),
    with cameras oriented to face toward the center and pitched slightly downward.
    
    Args:
        camtoworlds: Reference camera poses [N, 3, 4] or [N, 4, 4] to determine scene bounds
        n_frames: Number of frames in the trajectory
        height: Camera Z height. If None, uses mean height of reference cameras.
        pitch_degrees: Downward pitch angle in degrees (positive = looking down)
        wall_distance: Distance multiplier for wall placement (1.0=at camera bounding box, 
                       >1.0=outside cameras toward walls, <1.0=inside cameras)
        shape: Shape of the path ("rectangle" or "ellipse")
    
    Returns:
        Camera to world transforms [N, 4, 4]
    
    Note:
        Assumes Z axis is up (not the typical gsplat -Z up convention).
    """
    # Extract camera positions to determine scene bounds
    positions = camtoworlds[:, :3, 3]  # [N, 3]
    
    # Compute scene center and extents in XY plane
    center_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    center_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    center = np.array([center_x, center_y])
    
    extent_x = (positions[:, 0].max() - positions[:, 0].min()) / 2
    extent_y = (positions[:, 1].max() - positions[:, 1].min()) / 2
    
    # Print debug info
    print(f"\n[Wall Trajectory Debug]")
    print(f"  Reference cameras: {len(positions)}")
    print(f"  Center: ({center_x:.3f}, {center_y:.3f})")
    print(f"  Extent X: {extent_x:.3f}, Extent Y: {extent_y:.3f}")
    print(f"  Wall distance multiplier: {wall_distance}")
    print(f"  Trajectory radius X: {wall_distance * extent_x:.3f}")
    print(f"  Trajectory radius Y: {wall_distance * extent_y:.3f}")
    print(f"  Height: {height if height is not None else positions[:, 2].mean():.3f}")
    
    # Default height to mean of reference cameras
    if height is None:
        height = positions[:, 2].mean()
    
    # Generate path around perimeter
    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    
    if shape == "ellipse":
        # Elliptical path
        x = center[0] + wall_distance * extent_x * np.cos(t)
        y = center[1] + wall_distance * extent_y * np.sin(t)
    else:  # rectangle
        # Rectangular path - walk around the perimeter
        x = np.zeros(n_frames)
        y = np.zeros(n_frames)
        perimeter = 2 * (extent_x + extent_y) * wall_distance
        for i, ti in enumerate(t):
            # Normalize to [0, 4] for 4 sides
            side_pos = (ti / (2 * np.pi)) * 4
            if side_pos < 1:  # Right side (going up in Y)
                x[i] = center[0] + wall_distance * extent_x
                y[i] = center[1] - wall_distance * extent_y + 2 * wall_distance * extent_y * side_pos
            elif side_pos < 2:  # Top side (going left in X)
                x[i] = center[0] + wall_distance * extent_x - 2 * wall_distance * extent_x * (side_pos - 1)
                y[i] = center[1] + wall_distance * extent_y
            elif side_pos < 3:  # Left side (going down in Y)
                x[i] = center[0] - wall_distance * extent_x
                y[i] = center[1] + wall_distance * extent_y - 2 * wall_distance * extent_y * (side_pos - 2)
            else:  # Bottom side (going right in X)
                x[i] = center[0] - wall_distance * extent_x + 2 * wall_distance * extent_x * (side_pos - 3)
                y[i] = center[1] - wall_distance * extent_y
    
    z = np.full(n_frames, height)
    
    # Build camera-to-world transforms
    camtoworlds_out = np.zeros((n_frames, 4, 4))
    
    pitch_rad = np.radians(pitch_degrees)
    
    for i in range(n_frames):
        pos = np.array([x[i], y[i], z[i]])
        
        # Forward direction: point toward center (in XY plane)
        forward_xy = np.array([center[0] - x[i], center[1] - y[i]])
        forward_xy = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)
        
        # Initial forward (horizontal, pointing to center)
        forward = np.array([forward_xy[0], forward_xy[1], 0.0])
        
        # Right vector (perpendicular to forward in XY plane, Z is up)
        right = np.array([forward[1], -forward[0], 0.0])
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Apply pitch rotation around the right axis (positive pitch = look down)
        # Rotate forward vector down and up vector forward
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)
        
        # Rodrigues rotation formula around right axis
        up = np.array([0.0, 0.0, 1.0])
        forward_pitched = forward * cos_p + np.cross(right, forward) * sin_p
        up_pitched = up * cos_p + np.cross(right, up) * sin_p
        
        # Normalize
        forward_pitched = forward_pitched / (np.linalg.norm(forward_pitched) + 1e-8)
        up_pitched = up_pitched / (np.linalg.norm(up_pitched) + 1e-8)
        
        # Recompute right to ensure orthogonality
        right = np.cross(up_pitched, forward_pitched)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Build rotation matrix (columns are right, up, forward in camera coords)
        # OpenCV/COLMAP convention: camera looks along +Z, Y is down, X is right
        # So: R = [right | -up | forward] for camera-to-world
        R = np.column_stack([right, -up_pitched, forward_pitched])
        
        camtoworlds_out[i, :3, :3] = R
        camtoworlds_out[i, :3, 3] = pos
        camtoworlds_out[i, 3, 3] = 1.0
    
    return camtoworlds_out


def generate_elevated_trajectory(
    camtoworlds: np.ndarray,
    height: Optional[float] = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 1,
    subsample: int = 1,
    look_at_center: bool = True,
) -> np.ndarray:
    """
    Generate a trajectory that follows the original camera X,Y positions at a fixed height.
    
    Takes intermediate points from the original trajectory, keeps their X,Y coordinates,
    sets Z to the desired height, and orients cameras with the specified pitch.
    
    Args:
        camtoworlds: Reference camera poses [N, 3, 4] or [N, 4, 4]
        height: Camera Z height. If None, uses mean height of reference cameras.
        pitch_degrees: Downward pitch angle in degrees (positive = looking down)
        n_interp: Number of interpolation points between each original camera
        subsample: Use every nth original camera position (1 = use all)
        look_at_center: If True, cameras face toward scene center. If False, 
                        cameras face along the path direction.
    
    Returns:
        Camera to world transforms [N, 4, 4]
    
    Note:
        Assumes Z axis is up.
    """
    # Extract camera positions
    positions = camtoworlds[::subsample, :3, 3]  # [N, 3]
    n_cameras = len(positions)
    
    # Compute scene center for look-at direction
    center_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    center_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    center = np.array([center_x, center_y])
    
    # Default height to mean of reference cameras
    if height is None:
        height = positions[:, 2].mean()
    
    print(f"\n[Elevated Trajectory Debug]")
    print(f"  Original cameras: {len(camtoworlds)}, after subsample: {n_cameras}")
    print(f"  Scene center: ({center_x:.3f}, {center_y:.3f})")
    print(f"  Height: {height:.3f}")
    print(f"  Pitch: {pitch_degrees}°")
    print(f"  Interpolation: {n_interp} points between cameras")
    print(f"  Look at center: {look_at_center}")
    
    # Create positions at fixed height
    elevated_positions = positions.copy()
    elevated_positions[:, 2] = height
    
    # Interpolate between positions if requested
    if n_interp > 1:
        from scipy.interpolate import interp1d
        
        t_original = np.arange(n_cameras)
        t_interp = np.linspace(0, n_cameras - 1, n_cameras + (n_cameras - 1) * (n_interp - 1))
        
        interp_func = interp1d(t_original, elevated_positions, axis=0, kind='cubic')
        elevated_positions = interp_func(t_interp)
    
    n_frames = len(elevated_positions)
    print(f"  Total frames after interpolation: {n_frames}")
    
    # Build camera-to-world transforms
    camtoworlds_out = np.zeros((n_frames, 4, 4))
    pitch_rad = np.radians(pitch_degrees)
    
    for i in range(n_frames):
        pos = elevated_positions[i]
        
        if look_at_center:
            # Forward direction: point toward scene center (in XY plane)
            forward_xy = np.array([center[0] - pos[0], center[1] - pos[1]])
        else:
            # Forward direction: along the path
            if i < n_frames - 1:
                forward_xy = elevated_positions[i + 1, :2] - pos[:2]
            else:
                forward_xy = pos[:2] - elevated_positions[i - 1, :2]
        
        forward_xy = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)
        
        # Initial forward (horizontal)
        forward = np.array([forward_xy[0], forward_xy[1], 0.0])
        
        # Right vector (perpendicular to forward in XY plane, Z is up)
        right = np.array([forward[1], -forward[0], 0.0])
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Apply pitch rotation around the right axis (positive pitch = look down)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)
        
        # Rodrigues rotation formula around right axis
        up = np.array([0.0, 0.0, 1.0])
        forward_pitched = forward * cos_p + np.cross(right, forward) * sin_p
        up_pitched = up * cos_p + np.cross(right, up) * sin_p
        
        # Normalize
        forward_pitched = forward_pitched / (np.linalg.norm(forward_pitched) + 1e-8)
        up_pitched = up_pitched / (np.linalg.norm(up_pitched) + 1e-8)
        
        # Recompute right to ensure orthogonality
        right = np.cross(up_pitched, forward_pitched)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        # Build rotation matrix
        # OpenCV/COLMAP convention: camera looks along +Z, Y is down, X is right
        R = np.column_stack([right, -up_pitched, forward_pitched])
        
        camtoworlds_out[i, :3, :3] = R
        camtoworlds_out[i, :3, 3] = pos
        camtoworlds_out[i, 3, 3] = 1.0
    
    return camtoworlds_out


def generate_trajectory_from_selected_points(
    selected_positions: np.ndarray,
    height: Optional[float] = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 5,
    look_at_center: bool = True,
    closed_loop: bool = False,
) -> np.ndarray:
    """
    Generate a trajectory from user-selected points at a fixed height with pitch.
    
    Args:
        selected_positions: Selected XY(Z) positions [N, 2] or [N, 3]
        height: Camera Z height. If None, uses mean Z of selected points (or 0 if 2D).
        pitch_degrees: Downward pitch angle in degrees (positive = looking down)
        n_interp: Number of interpolation points between each selected point
        look_at_center: If True, cameras face toward centroid. If False, face along path.
        closed_loop: If True, connect last point back to first.
    
    Returns:
        Camera to world transforms [N, 4, 4]
    """
    positions = np.array(selected_positions)
    n_selected = len(positions)
    
    if n_selected < 2:
        raise ValueError("Need at least 2 selected points to generate trajectory")
    
    # Handle 2D vs 3D input
    if positions.shape[1] == 2:
        # Add Z coordinate
        if height is None:
            height = 0.0
        positions = np.column_stack([positions, np.full(n_selected, height)])
    elif height is not None:
        # Override Z with specified height
        positions[:, 2] = height
    
    # Compute center for look-at
    center = positions[:, :2].mean(axis=0)
    
    print(f"\n[Selected Points Trajectory]")
    print(f"  Selected points: {n_selected}")
    print(f"  Center: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"  Height: {positions[0, 2]:.3f}")
    print(f"  Pitch: {pitch_degrees}°")
    print(f"  Interpolation: {n_interp} points between keypoints")
    print(f"  Closed loop: {closed_loop}")
    
    # Add first point at end for closed loop
    if closed_loop:
        positions = np.vstack([positions, positions[0:1]])
    
    # Interpolate between points
    if n_interp > 1:
        from scipy.interpolate import interp1d, CubicSpline
        
        # For closed loops, sort keypoints by angle around the center so the
        # spline follows the circle consistently in XY (ignoring click order).
        if closed_loop:
            xy = positions[:, :2]
            rel = xy - center  # center computed earlier from XY positions
            angles = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort(angles)
            positions = positions[order]
        
        n_key = len(positions)
        t_original = np.linspace(0.0, 1.0, n_key)
        n_total = n_key + (n_key - 1) * (n_interp - 1)
        t_interp = np.linspace(0.0, 1.0, n_total)
        
        if closed_loop and n_key >= 3:
            # Use a periodic cubic spline for smooth closed loops in XY.
            spline = CubicSpline(t_original, positions, axis=0, bc_type="periodic")
            positions = spline(t_interp)
        else:
            # Standard cubic interpolation for open paths
            interp_func = interp1d(t_original, positions, axis=0, kind="cubic")
            positions = interp_func(t_interp)
    
    n_frames = len(positions)
    print(f"  Total frames: {n_frames}")
    
    # Build camera-to-world transforms
    camtoworlds_out = np.zeros((n_frames, 4, 4))
    pitch_rad = np.radians(pitch_degrees)
    
    for i in range(n_frames):
        pos = positions[i]
        
        if look_at_center:
            forward_xy = center - pos[:2]
        else:
            if i < n_frames - 1:
                forward_xy = positions[i + 1, :2] - pos[:2]
            else:
                forward_xy = pos[:2] - positions[i - 1, :2]
        
        forward_xy = forward_xy / (np.linalg.norm(forward_xy) + 1e-8)
        forward = np.array([forward_xy[0], forward_xy[1], 0.0])
        
        right = np.array([forward[1], -forward[0], 0.0])
        right = right / (np.linalg.norm(right) + 1e-8)
        
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)
        
        up = np.array([0.0, 0.0, 1.0])
        forward_pitched = forward * cos_p + np.cross(right, forward) * sin_p
        up_pitched = up * cos_p + np.cross(right, up) * sin_p
        
        forward_pitched = forward_pitched / (np.linalg.norm(forward_pitched) + 1e-8)
        up_pitched = up_pitched / (np.linalg.norm(up_pitched) + 1e-8)
        
        right = np.cross(up_pitched, forward_pitched)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        R = np.column_stack([right, -up_pitched, forward_pitched])
        
        camtoworlds_out[i, :3, :3] = R
        camtoworlds_out[i, :3, 3] = pos
        camtoworlds_out[i, 3, 3] = 1.0
    
    return camtoworlds_out


def interactive_pick_trajectory(
    data_dir: Path,
    height: Optional[float] = None,
    pitch_degrees: float = 15.0,
    n_interp: int = 5,
    look_at_center: bool = True,
    closed_loop: bool = False,
    output_path: Optional[Path] = None,
):
    """
    Interactive mode to pick ANY 3D points and generate a trajectory.
    
    Opens a visualization where you can:
    1. Shift+Click on ANY point in the scene to select trajectory waypoints
    2. Select points in the ORDER you want the trajectory to follow
    3. Press 'Q' to finish selection and generate trajectory
    4. View the generated trajectory
    
    Args:
        data_dir: Path to COLMAP dataset directory
        height: Camera Z height for trajectory. None uses picked point heights.
        pitch_degrees: Downward pitch angle in degrees
        n_interp: Interpolation points between selected points
        look_at_center: If True, cameras face toward centroid
        closed_loop: If True, connect last point back to first
        output_path: If provided, save the trajectory to this file (.npy format)
    """
    import open3d as o3d
    
    print(f"Loading COLMAP data from {data_dir}...")
    points, colors, camtoworlds_original, scene_scale = colmap_parser(data_dir)
    
    cam_positions = camtoworlds_original[:, :3, 3]
    n_scene_points = len(points)
    n_cameras = len(cam_positions)
    
    print(f"Loaded {n_scene_points} scene points, {n_cameras} cameras")
    
    # Dump original camera trajectory for debugging
    baseline_path = Path(data_dir) / "baseline.npy"
    np.save(baseline_path, camtoworlds_original)
    print(f"Saved baseline camera trajectory to {baseline_path}")
    
    # Print bounds to help user understand the scene
    print(f"\nScene bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Create point cloud with scene points only
    # Camera positions shown in green for reference
    combined_points = np.vstack([points, cam_positions])
    cam_colors = np.array([[0.0, 1.0, 0.0]] * n_cameras)  # Green for cameras
    combined_colors = np.vstack([colors, cam_colors])
    
    # Calculate scene center and extent for coordinate frame
    scene_center = combined_points.mean(axis=0)
    scene_extent = combined_points.max(axis=0) - combined_points.min(axis=0)
    frame_size = max(scene_extent) * 0.15  # 15% of scene extent
    
    # Add coordinate frame as colored points directly in the point cloud
    # Create points along each axis (X=red, Y=green, Z=blue)
    n_axis_points = 20  # Number of points per axis
    axis_points_list = []
    axis_colors_list = []
    
    # X axis (red) - from center along +X
    for i in range(n_axis_points + 1):
        t = i / n_axis_points
        axis_points_list.append(scene_center + [t * frame_size, 0, 0])
        axis_colors_list.append([1.0, 0.0, 0.0])  # Red
    
    # Y axis (green) - from center along +Y
    for i in range(n_axis_points + 1):
        t = i / n_axis_points
        axis_points_list.append(scene_center + [0, t * frame_size, 0])
        axis_colors_list.append([0.0, 1.0, 0.0])  # Green
    
    # Z axis (blue) - from center along +Z
    for i in range(n_axis_points + 1):
        t = i / n_axis_points
        axis_points_list.append(scene_center + [0, 0, t * frame_size])
        axis_colors_list.append([0.0, 0.0, 1.0])  # Blue
    
    # Add axis points to combined point cloud
    axis_points_array = np.array(axis_points_list)
    axis_colors_array = np.array(axis_colors_list)
    combined_points = np.vstack([combined_points, axis_points_array])
    combined_colors = np.vstack([combined_colors, axis_colors_array])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    print(f"\n{'='*60}")
    print("INTERACTIVE POINT SELECTION")
    print(f"{'='*60}")
    print("Instructions:")
    print("  1. Hold SHIFT and LEFT-CLICK on ANY point to select waypoints")
    print("  2. Click points in the ORDER you want the trajectory to follow")
    print("  3. Pick points along walls/edges where you want the camera path")
    print("  4. Press 'Q' when done to close window and generate trajectory")
    print(f"\nTips:")
    print(f"  - GREEN points = original camera positions (for reference)")
    print(f"  - You can click on scene points OR camera points")
    print(f"  - Height will be set to: {height if height is not None else 'picked point Z coords'}")
    print(f"{'='*60}\n")
    
    # Use VisualizerWithVertexSelection for picking
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Pick Trajectory Waypoints (Shift+Click anywhere)", width=1280, height=720)
    
    vis.add_geometry(pcd)
    
    print(f"\nCoordinate frame at scene center: {scene_center}")
    print(f"  X axis (RED):   {scene_center} -> {scene_center + [frame_size, 0, 0]}")
    print(f"  Y axis (GREEN): {scene_center} -> {scene_center + [0, frame_size, 0]}")
    print(f"  Z axis (BLUE):  {scene_center} -> {scene_center + [0, 0, frame_size]}")
    print(f"  Look for bright RED, GREEN, and BLUE lines in the point cloud!")
    
    vis.run()
    
    # Get picked points - returns list of PickedPoint objects
    picked_points = vis.get_picked_points()
    vis.destroy_window()
    
    if len(picked_points) < 2:
        print(f"ERROR: Only {len(picked_points)} points selected. Need at least 2.")
        return None
    
    # Extract indices from PickedPoint objects
    # PickedPoint has .index and .coord attributes
    picked_indices = []
    for pp in picked_points:
        if hasattr(pp, 'index'):
            picked_indices.append(pp.index)
        else:
            # Fallback: might be just an integer in some Open3D versions
            picked_indices.append(int(pp))
    
    # Get positions for ALL picked points (scene points or camera points)
    selected_positions = []
    
    print(f"\nSelected {len(picked_indices)} waypoints:")
    for i, idx in enumerate(picked_indices):
        pos = combined_points[idx]
        selected_positions.append(pos)
        point_type = "camera" if idx >= n_scene_points else "scene"
        print(f"  {i+1}. [{point_type}] ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    selected_positions = np.array(selected_positions)
    
    # Generate trajectory from selected points
    trajectory = None
    # trajectory = generate_trajectory_from_selected_points(
    #     selected_positions,
    #     height=height,
    #     pitch_degrees=pitch_degrees,
    #     n_interp=n_interp,
    #     look_at_center=look_at_center,
    #     closed_loop=closed_loop,
    # )
    
    # Save trajectory
    if output_path is not None:
        output_path = Path(output_path)
        if trajectory is not None:
            np.save(output_path, trajectory)
            print(f"\n{'='*60}")
            print(f"TRAJECTORY SAVED")
            print(f"{'='*60}")
            print(f"  File: {output_path}")
            print(f"  Poses: {len(trajectory)}")
            print(f"  Shape: {trajectory.shape}")
            print(f"\nTo render video with generate_traj.py:")
            print(f"  python generate_traj.py --data-dir {data_dir} --ckpt <checkpoint.pt> \\")
            print(f"      --traj-file {output_path} -o output.mp4")
            print(f"{'='*60}")
            print(f"\nSaved trajectory to {output_path}")
    
    # Visualize the result
    print("\nShowing generated trajectory...")
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Generated Trajectory", width=1280, height=720)
    
    vis2.add_geometry(pcd)
    
    # Show original cameras in green
    for pos in cam_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(pos)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        vis2.add_geometry(sphere)
    
    # Show selected points in yellow (larger)
    for pos in selected_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sphere.translate(pos)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        vis2.add_geometry(sphere)
    
    # Show generated trajectory in red
    if trajectory is not None:
        traj_positions = trajectory[:, :3, 3]
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(traj_positions)
        lines = [[i, i+1] for i in range(len(traj_positions)-1)]
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(lines))
        vis2.add_geometry(lineset)
    
        # Show trajectory camera positions
        for i, pos in enumerate(traj_positions[::5]):  # Every 5th for clarity
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(pos)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            vis2.add_geometry(sphere)
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis2.add_geometry(coord_frame)
    
    print("\nVisualization colors:")
    print("  GREEN  = Original camera positions")
    print("  YELLOW = Selected keypoints")
    print("  RED    = Generated trajectory")
    
    vis2.run()
    vis2.destroy_window()
    
    return trajectory


def convert_zup_to_colmap(camtoworlds: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world matrices from Z-up convention to COLMAP/OpenCV convention.
    
    COLMAP uses: X right, Y down, Z forward
    Z-up uses: X right, Y forward, Z up
    
    This function transforms trajectories generated with Z-up assumption to work
    with the COLMAP-based renderer.
    
    Args:
        camtoworlds: Camera to world transforms [N, 4, 4] in Z-up convention
    
    Returns:
        Camera to world transforms [N, 4, 4] in COLMAP convention
    """
    # Transform matrix: Z-up -> COLMAP
    # COLMAP X = Z-up X
    # COLMAP Y = -Z-up Z (down is negative up)
    # COLMAP Z = Z-up Y
    transform = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1],
    ], dtype=np.float32)
    
    result = camtoworlds.copy()
    for i in range(len(result)):
        # Transform: new_c2w = transform @ old_c2w @ transform^-1
        # But we need to transform both position and rotation
        # Position transform
        pos = result[i, :3, 3]
        pos_transformed = transform[:3, :3] @ pos
        result[i, :3, 3] = pos_transformed
        
        # Rotation transform: R_new = transform[:3, :3] @ R_old @ transform[:3, :3].T
        rot = result[i, :3, :3]
        rot_transformed = transform[:3, :3] @ rot @ transform[:3, :3].T
        result[i, :3, :3] = rot_transformed
    
    return result


def load_trajectory(path: Path, convert_zup: bool = True) -> np.ndarray:
    """
    Load trajectory from file.
    
    Args:
        path: Path to trajectory file (.npy or .npz)
        convert_zup: If True, convert from Z-up to COLMAP convention
    
    Returns:
        Camera to world transforms [N, 4, 4]
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        camtoworlds = data["camtoworlds"]
    else:
        camtoworlds = np.load(path)
    
    # Convert from Z-up to COLMAP convention if requested
    if convert_zup:
        camtoworlds = convert_zup_to_colmap(camtoworlds)
    
    return camtoworlds


def offset_trajectory(
    camtoworlds: np.ndarray, 
    offset: np.ndarray
) -> np.ndarray:
    """
    Apply a world-space offset to camera positions.
    
    Args:
        camtoworlds: Camera to world transforms [N, 4, 4] or [N, 3, 4]
        offset: World-space offset [3,] (x, y, z)
    
    Returns:
        Modified camera transforms with offset applied to positions.
        
    Note:
        In typical gsplat/NeRF setups, up is -Z direction, so to move
        the camera "up" by 10cm, use offset=(0, 0, -0.1).
    """
    result = camtoworlds.copy()
    result[..., :3, 3] += offset
    return result

    
def transform_trajectory(
    camtoworlds: np.ndarray,
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply a general transformation to camera trajectory.

    Args:
        camtoworlds: Camera to world transforms [N, 4, 4]
        rotation: 3x3 rotation matrix to apply (in world space)
        translation: 3D translation vector to apply (in world space)
        scale: Scale factor to apply to positions

    Returns:
        Transformed camera to world matrices [N, 4, 4]
    """
    result = camtoworlds.copy()

    # Scale positions
    if scale != 1.0:
        result[..., :3, 3] *= scale

    # Apply rotation (rotates camera positions and orientations)
    if rotation is not None:
        for i in range(len(result)):
            # Rotate position
            result[i, :3, 3] = rotation @ result[i, :3, 3]
            # Rotate orientation
            result[i, :3, :3] = rotation @ result[i, :3, :3]

    # Apply translation
    if translation is not None:
        result[..., :3, 3] += translation

    return result


def apply_elevation_and_pitch(
    camtoworlds: np.ndarray,
    elevation: float = 0.0,
    pitch_degrees: float = 0.0,
) -> np.ndarray:
    """
    Apply elevation (Z-axis translation) and pitch rotation to camera trajectory.

    This function modifies the camera poses by:
    1. Translating all cameras upward by the elevation amount (in Z direction)
    2. Rotating each camera by pitch_degrees around its own right axis (camera space)

    Args:
        camtoworlds: Camera to world transforms [N, 4, 4] or [N, 3, 4]
                    Input should be in world space where +Z is up
        elevation: Amount to raise cameras in Z direction (positive = up)
        pitch_degrees: Pitch angle in degrees to tilt camera around its right axis
                      (positive = tilt camera down / look down)

    Returns:
        Transformed camera to world matrices [N, 4, 4]

    Example:
        # Raise cameras by 0.5 units and tilt down by 15 degrees
        camtoworlds_modified = apply_elevation_and_pitch(
            camtoworlds_all,
            elevation=0.5,
            pitch_degrees=15.0
        )

    Note:
        Assumes Z axis is up (world space convention).
        Pitch is applied in camera space around each camera's right axis,
        where +Z forward is the camera's viewing direction.
    """
    # Ensure 4x4 format
    result = ensure_4x4(camtoworlds)

    # Apply elevation (Z translation)
    if elevation != 0.0:
        translation = np.array([0.0, 0.0, elevation])
        result[..., :3, 3] += translation
        print(f"Applied elevation: +{elevation:.3f} in Z")

    # Apply pitch rotation in camera space
    if pitch_degrees != 0.0:
        pitch_rad = np.radians(pitch_degrees)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)

        # Apply rotation to each camera in its own camera space
        for i in range(len(result)):
            # Extract camera rotation matrix (camtoworld)
            R = result[i, :3, :3]

            # Camera's right axis is the first column of R (X-axis in camera space)
            right = R[:, 0]

            # Build rotation matrix around the camera's right axis using Rodrigues formula
            # R(axis, angle) = I + sin(angle)*K + (1-cos(angle))*K^2
            # where K is the skew-symmetric matrix of the axis
            K = np.array([
                [0, -right[2], right[1]],
                [right[2], 0, -right[0]],
                [-right[1], right[0], 0]
            ])

            # Rodrigues rotation formula
            pitch_rotation = np.eye(3) + sin_p * K + (1 - cos_p) * (K @ K)

            # Apply pitch rotation to the camera orientation
            # New orientation = pitch_rotation @ old_orientation
            result[i, :3, :3] = pitch_rotation @ R

        print(f"Applied pitch rotation: {pitch_degrees:.1f}° around each camera's right axis")

    return result


def ensure_4x4(camtoworlds: np.ndarray) -> np.ndarray:
    """
    Ensure camera matrices are 4x4 by adding homogeneous row if needed.
    
    Args:
        camtoworlds: Camera matrices [N, 3, 4] or [N, 4, 4]
    
    Returns:
        Camera matrices [N, 4, 4]
    """
    if camtoworlds.shape[1] == 3:
        # Add homogeneous row [0, 0, 0, 1]
        homogeneous = np.repeat(
            np.array([[[0.0, 0.0, 0.0, 1.0]]]), 
            len(camtoworlds), 
            axis=0
        )
        camtoworlds = np.concatenate([camtoworlds, homogeneous], axis=1)
    return camtoworlds


def generate_trajectory_from_parser(
    parser: Parser,
    traj_type: Literal["interp", "ellipse", "spiral", "wall", "elevated", "colmap"] = "interp",
    n_frames: Optional[int] = None,
    scene_scale: float = 1.0,
    elevation: Optional[float] = None,
    pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: Literal["rectangle", "ellipse"] = "ellipse",
    elevated_subsample: int = 1,
    elevated_look_at_center: bool = True,
    colmap_interp: int = 1,
) -> np.ndarray:
    """
    Generate a camera trajectory from a Parser, replicating render_traj logic.
    
    Args:
        parser: COLMAP Parser with camera poses
        traj_type: Type of trajectory ("interp", "ellipse", "spiral", "wall", "elevated")
        n_frames: Number of frames (None uses defaults based on traj_type)
        scene_scale: Scene scale factor (used for spiral trajectory)
        wall_height: Camera height for wall/elevated trajectory (Z coordinate). None uses mean.
        wall_pitch_degrees: Downward pitch for wall/elevated trajectory (degrees)
        wall_distance: How close to walls (0=center, 1=at edge) for wall trajectory
        wall_shape: Path shape for wall trajectory ("rectangle" or "ellipse")
        elevated_subsample: Use every nth camera for elevated trajectory
        elevated_look_at_center: If True, elevated cameras face scene center
        colmap_interp: Interpolation factor for colmap trajectory (1 = use original poses)
    
    Returns:
        Camera to world transforms [N, 4, 4]
    """
    # Use poses excluding first and last 5 (same as render_traj)
    camtoworlds_all = parser.camtoworlds[5:-5]
    
    if traj_type == "interp":
        # Interpolated path through keyframes
        n_interp = 1 if n_frames is None else max(1, n_frames // len(camtoworlds_all))
        camtoworlds_all = generate_interpolated_path(camtoworlds_all, n_interp)
        camtoworlds_all = apply_elevation_and_pitch(camtoworlds_all, elevation=elevation, pitch_degrees=pitch_degrees)
    elif traj_type == "ellipse":
        # Elliptical path at average height
        height = camtoworlds_all[:, 2, 3].mean()
        n = 120 if n_frames is None else n_frames
        camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, n_frames=n, height=height)
    elif traj_type == "spiral":
        # Spiral path (for forward-facing scenes)
        n = 120 if n_frames is None else n_frames
        camtoworlds_all = generate_spiral_path(
            camtoworlds_all,
            bounds=parser.bounds * scene_scale,
            n_frames=n,
            spiral_scale_r=parser.extconf.get("spiral_radius_scale", 1.0),
        )
    elif traj_type == "wall":
        # Wall trajectory - around room perimeter facing inward
        n = 120 if n_frames is None else n_frames
        camtoworlds_all = generate_wall_trajectory(
            camtoworlds_all,
            n_frames=n,
            height=elevation,
            pitch_degrees=pitch_degrees,
            wall_distance=wall_distance,
            shape=wall_shape,
        )
    elif traj_type == "elevated":
        # Elevated trajectory - follows original X,Y at fixed height with pitch
        n_interp = 1 if n_frames is None else max(1, n_frames // (len(camtoworlds_all) // elevated_subsample))
        camtoworlds_all = generate_elevated_trajectory(
            camtoworlds_all,
            height=elevation,
            pitch_degrees=pitch_degrees,
            n_interp=n_interp,
            subsample=elevated_subsample,
            look_at_center=elevated_look_at_center,
        )
    elif traj_type == "colmap":
        # Use original COLMAP camera poses directly (all poses, not trimmed)
        camtoworlds_all = parser.camtoworlds.copy()
        if colmap_interp > 1:
            # Optionally interpolate between original poses
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, colmap_interp)
        print(f"Using original COLMAP camera poses: {len(camtoworlds_all)} poses")
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    
    # Ensure 4x4 format
    camtoworlds_all = ensure_4x4(camtoworlds_all)
    
    return camtoworlds_all


def generate_render_traj_offset_up(
    parser: Parser,
    traj_type: Literal["interp", "ellipse", "spiral", "wall", "elevated", "colmap"] = "interp",
    up_offset_meters: float = 0.1,
    scene_scale: float = 1.0,
    n_frames: Optional[int] = None,
    wall_height: Optional[float] = None,
    wall_pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: Literal["rectangle", "ellipse"] = "ellipse",
    elevated_subsample: int = 1,
    elevated_look_at_center: bool = True,
    colmap_interp: int = 1,
) -> np.ndarray:
    """
    Generate a trajectory like render_traj but with camera moved up.
    
    This is a convenience function that generates the same trajectory
    as render_traj in simple_trainer.py, but offsets the camera position
    upward by the specified amount.
    
    Args:
        parser: COLMAP Parser with camera poses
        traj_type: Type of trajectory ("interp", "ellipse", "spiral", "wall", "elevated", "colmap")
        up_offset_meters: How much to move camera up in meters (positive = up)
        scene_scale: Scene scale factor (important if scene is normalized!)
        n_frames: Number of frames (None uses defaults)
        wall_height: Camera height for wall/elevated trajectory
        wall_pitch_degrees: Downward pitch for wall/elevated trajectory
        wall_distance: How close to walls for wall trajectory
        wall_shape: Path shape for wall trajectory
        elevated_subsample: Use every nth camera for elevated trajectory
        elevated_look_at_center: If True, elevated cameras face scene center
        colmap_interp: Interpolation factor for colmap trajectory (1 = use original poses)
    
    Returns:
        Camera to world transforms [N, 4, 4] with upward offset applied
    
    Note:
        In gsplat, up direction is typically (0, 0, -1), so moving "up" 
        means decreasing Z. If the scene is normalized, you may need to
        adjust up_offset_meters by the scene_scale.
    """
    # Generate base trajectory
    camtoworlds = generate_trajectory_from_parser(
        parser, 
        traj_type=traj_type, 
        n_frames=n_frames,
        scene_scale=scene_scale,
        elevation=wall_height,
        pitch_degrees=wall_pitch_degrees,
        wall_distance=wall_distance,
        wall_shape=wall_shape,
        elevated_subsample=elevated_subsample,
        elevated_look_at_center=elevated_look_at_center,
        colmap_interp=colmap_interp,
    )
    
    # Apply upward offset (up is -Z in gsplat coordinate system)
    # The offset needs to be in the same scale as the scene
    scaled_offset = up_offset_meters / scene_scale if scene_scale != 0 else up_offset_meters
    offset = np.array([0.0, 0.0, -scaled_offset])  # -Z is up
    
    camtoworlds = offset_trajectory(camtoworlds, offset)
    
    return camtoworlds


def setup_image_directory(data_dir: Path, factor: int = 1) -> None:
    """
    Set up image directory structure expected by COLMAP parser.
    
    Looks for images named like front0, front1, etc. and creates
    the expected directory structure (images/ or images_4/).
    
    Args:
        data_dir: Path to COLMAP dataset directory
        factor: Downsample factor (determines if images_4 is needed)
    """
    import os
    import shutil
    import glob
    
    data_dir = Path(data_dir)
    
    # Check what image directory is expected
    if factor > 1:
        image_dir = data_dir / f"images_{factor}"
        base_image_dir = data_dir / "images"
    else:
        image_dir = data_dir / "images"
        base_image_dir = None
    
    # If directory already exists and has files, we're done
    if image_dir.exists():
        existing_files = list(image_dir.glob("*"))
        # Filter out directories and hidden files
        existing_files = [f for f in existing_files if f.is_file() and not f.name.startswith('.')]
        if len(existing_files) > 0:
            return
        # Directory exists but is empty - we'll try to populate it
    
    # Look for images in the data_dir root (front0, front1, etc.)
    image_patterns = [
        "front*",
        "image*",
        "*.jpg",
        "*.png",
        "*.JPG",
        "*.PNG",
    ]
    
    found_images = []
    for pattern in image_patterns:
        found_images.extend(glob.glob(str(data_dir / pattern)))
        if found_images:
            break
    
    if not found_images:
        # Try looking in common subdirectories
        for subdir in ["images", "raw", "input"]:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                for pattern in image_patterns:
                    found_images.extend(glob.glob(str(subdir_path / pattern)))
                    if found_images:
                        break
                if found_images:
                    break
    
    # Also check if base images/ directory has files we can use
    base_images_dir = data_dir / "images"
    if not found_images and base_images_dir.exists():
        base_images = list(base_images_dir.glob("*"))
        base_images = [f for f in base_images if f.is_file() and not f.name.startswith('.')]
        if base_images:
            found_images = [str(f) for f in base_images]
    
    if found_images:
        print(f"Found {len(found_images)} images, setting up {image_dir}...")
        image_dir.mkdir(exist_ok=True)
        
        # Copy or symlink images
        for img_path in found_images:
            img_name = Path(img_path).name
            dest = image_dir / img_name
            if not dest.exists():
                try:
                    os.symlink(img_path, dest)
                except OSError:
                    # Fallback to copy if symlink fails (e.g., on Windows)
                    shutil.copy2(img_path, dest)
        
        # Also create base images/ directory if needed
        if base_image_dir and not base_image_dir.exists():
            base_image_dir.mkdir(exist_ok=True)
            for img_path in found_images:
                img_name = Path(img_path).name
                dest = base_image_dir / img_name
                if not dest.exists():
                    try:
                        os.symlink(img_path, dest)
                    except OSError:
                        shutil.copy2(img_path, dest)
        
        print(f"Created {image_dir} with {len(found_images)} images")
    else:
        # Create empty directory as fallback (parser will fail later but at least we tried)
        print(f"Warning: No images found. Creating empty {image_dir} directory.")
        print(f"  You may need to manually create {image_dir} and add images.")
        image_dir.mkdir(exist_ok=True)
        if base_image_dir:
            base_image_dir.mkdir(exist_ok=True)


def colmap_parser(data_dir: Path, factor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load point cloud and camera poses from a COLMAP dataset.
    
    Args:
        data_dir: Path to COLMAP dataset directory (containing sparse/0 or sparse/)
        factor: Downsample factor for images. If None, auto-detects from existing directories.
    
    Returns:
        Tuple of:
            - points: Point cloud positions [N, 3]
            - colors: Point cloud colors [N, 3] (0-1 range)
            - camtoworlds: Camera to world transforms [M, 4, 4]
            - scene_scale: Scene scale factor
    """
    data_dir = Path(data_dir)
    
    # Auto-detect factor if not provided
    if factor is None:
        # Default to full resolution (factor=1)
        factor = 1
        print(f"Using full resolution images (factor=1)")
    
    # Try to set up image directory if it doesn't exist
    setup_image_directory(data_dir, factor)
    
    parser = Parser(
        data_dir=str(data_dir),
        factor=factor,
        normalize=True,
        test_every=8,
    )
    
    points = parser.points  # [N, 3]
    colors = parser.points_rgb / 255.0  # [N, 3] normalized to 0-1
    camtoworlds = parser.camtoworlds  # [M, 4, 4]
    scene_scale = parser.scene_scale * 1.1
    
    return points, colors, camtoworlds, scene_scale


def create_trajectory_lineset(
    camtoworlds: np.ndarray,
    color: List[float] = [0.0, 0.0, 1.0],
    subsample: int = 1,
) -> "o3d.geometry.LineSet":
    """
    Create an Open3D LineSet from camera trajectory.
    
    Args:
        camtoworlds: Camera to world transforms [N, 4, 4]
        color: RGB color for the trajectory line
        subsample: Only use every nth pose for the line
    
    Returns:
        Open3D LineSet geometry
    """
    import open3d as o3d
    
    # Extract camera positions
    positions = camtoworlds[::subsample, :3, 3]  # [N, 3]
    
    # Create line segments connecting consecutive positions
    n_points = len(positions)
    lines = [[i, i + 1] for i in range(n_points - 1)]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(positions)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return lineset


def create_trajectory_spheres(
    camtoworlds: np.ndarray,
    color: List[float] = [0.0, 0.0, 1.0],
    radius: float = 0.02,
    subsample: int = 10,
) -> List["o3d.geometry.TriangleMesh"]:
    """
    Create spheres at camera positions for trajectory visualization.
    
    Args:
        camtoworlds: Camera to world transforms [N, 4, 4]
        color: RGB color for the spheres
        radius: Sphere radius
        subsample: Only create sphere for every nth pose
    
    Returns:
        List of Open3D sphere meshes
    """
    import open3d as o3d
    
    positions = camtoworlds[::subsample, :3, 3]
    spheres = []
    
    for pos in positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    
    return spheres


def view_trajectory(
    data_dir: Path,
    traj_before: Optional[np.ndarray] = None,
    traj_after: Optional[np.ndarray] = None,
    up_offset_meters: float = 0.1,
    traj_type: Literal["interp", "ellipse", "spiral", "wall", "elevated"] = "interp",
    point_size: float = 2.0,
    show_original_cameras: bool = True,
    wall_height: Optional[float] = None,
    wall_pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: Literal["rectangle", "ellipse"] = "ellipse",
    n_frames: Optional[int] = None,
    elevated_subsample: int = 1,
    elevated_look_at_center: bool = True,
):
    """
    Visualize COLMAP point cloud with before/after trajectories using Open3D.
    
    Args:
        data_dir: Path to COLMAP dataset directory
        traj_before: Original trajectory [N, 4, 4] (blue). If None, generates from parser.
        traj_after: Offset trajectory [N, 4, 4] (red). If None, generates with up_offset.
        up_offset_meters: Offset to apply for traj_after if not provided
        traj_type: Trajectory type to generate if traj_before/after not provided
        point_size: Size of points in visualization
        show_original_cameras: Whether to show the original COLMAP camera positions (green)
        wall_height: Camera height for wall trajectory (Z coordinate)
        wall_pitch_degrees: Downward pitch for wall trajectory (degrees)
        wall_distance: How close to walls (0=center, 1=at edge)
        wall_shape: Path shape for wall trajectory
        n_frames: Number of frames in trajectory
    """
    import open3d as o3d
    
    # Load point cloud and camera data
    print(f"Loading COLMAP data from {data_dir}...")
    points, colors, camtoworlds_original, scene_scale = colmap_parser(data_dir)
    
    print(f"Loaded {len(points)} points, {len(camtoworlds_original)} cameras")
    print(f"Scene scale: {scene_scale:.4f}")
    
    # Print bounding box dimensions
    cam_positions = camtoworlds_original[:, :3, 3]
    print(f"\nPoint cloud bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"Camera position bounds:")
    print(f"  X: [{cam_positions[:, 0].min():.3f}, {cam_positions[:, 0].max():.3f}]")
    print(f"  Y: [{cam_positions[:, 1].min():.3f}, {cam_positions[:, 1].max():.3f}]")
    print(f"  Z: [{cam_positions[:, 2].min():.3f}, {cam_positions[:, 2].max():.3f}]")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Generate trajectories if not provided
    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=8)
    
    if traj_before is None:
        print(f"[BEFORE] Generating interpolated original camera trajectory...")
        traj_before = generate_trajectory_from_parser(
            parser, traj_type="interp", scene_scale=scene_scale,
            n_frames=n_frames,
        )
    
    if traj_after is None:
        print(f"[AFTER] Generating '{traj_type}' trajectory...")
        traj_after = generate_trajectory_from_parser(
            parser, 
            traj_type=traj_type, 
            scene_scale=scene_scale,
            n_frames=n_frames,
            elevation=wall_height,
            pitch_degrees=wall_pitch_degrees,
            wall_distance=wall_distance,
            wall_shape=wall_shape,
            elevated_subsample=elevated_subsample,
            elevated_look_at_center=elevated_look_at_center,
        )
    
    print(f"[BEFORE] Interpolated trajectory: {len(traj_before)} poses")
    print(f"[AFTER] {traj_type.capitalize()} trajectory: {len(traj_after)} poses")
    
    # Subsample for visualization (too many points can be slow)
    subsample = max(1, len(traj_before) // 200)
    
    # Create trajectory visualizations
    # Before trajectory: BLUE lines and spheres
    lineset_before = create_trajectory_lineset(traj_before, color=[0.0, 0.0, 1.0], subsample=subsample)
    geometries.append(lineset_before)
    
    spheres_before = create_trajectory_spheres(
        traj_before, color=[0.0, 0.0, 1.0], radius=0.015, subsample=subsample * 5
    )
    geometries.extend(spheres_before)
    
    # After trajectory: RED lines and spheres
    lineset_after = create_trajectory_lineset(traj_after, color=[1.0, 0.0, 0.0], subsample=subsample)
    geometries.append(lineset_after)
    
    spheres_after = create_trajectory_spheres(
        traj_after, color=[1.0, 0.0, 0.0], radius=0.015, subsample=subsample * 5
    )
    geometries.extend(spheres_after)
    
    # Original COLMAP camera positions: GREEN spheres
    if show_original_cameras:
        spheres_original = create_trajectory_spheres(
            camtoworlds_original, color=[0.0, 1.0, 0.0], radius=0.01, subsample=1
        )
        geometries.extend(spheres_original)
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    # Visualize
    print("\nVisualization colors:")
    print("  BLUE  = Original trajectory (before offset)")
    print("  RED   = Offset trajectory (after offset)")
    if show_original_cameras:
        print("  GREEN = Original COLMAP camera positions")
    print("\nControls: Left-click drag to rotate, scroll to zoom, right-click drag to pan")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Trajectory Visualization",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def save_baseline_trajectory(
    data_dir: Path,
    output_path: Optional[Path] = None,
    transform_path: Optional[Path] = None,
    factor: int = 1,
) -> Path:
    """
    Save baseline.npy with original COLMAP camera poses.
    
    Args:
        data_dir: Path to COLMAP dataset directory
        output_path: Path to save baseline.npy. If None, saves to data_dir/baseline.npy
        factor: Downsample factor for images (default: 1 for full resolution)
    
    Returns:
        Path to saved baseline.npy file
    
    Note:
        baseline.npy is saved in COLMAP convention (Y-down). When loaded with
        load_trajectory(), use convert_zup=False (default) to keep COLMAP convention.
    """
    data_dir = Path(data_dir)
    
    # Set up image directory if needed
    setup_image_directory(data_dir, factor)
    
    # Load COLMAP parser
    parser = Parser(
        data_dir=str(data_dir),
        factor=factor,
        normalize=True,
        test_every=8,
    )
    
    # Determine output path
    if output_path is None:
        output_path = data_dir / "baseline.npy"
    else:
        output_path = Path(output_path)
        
    if transform_path is None:
        transform_path = data_dir / "transform.npy"
    
    # Save original COLMAP camera poses (already in COLMAP convention)
    np.save(output_path, parser.camtoworlds)
    np.save(transform_path, parser.transform)
    print(f"Saved baseline camera trajectory ({len(parser.camtoworlds)} poses) to {output_path}")
    print("  Note: baseline.npy is in COLMAP convention (Y-down). When loaded with")
    print("  load_trajectory(), use convert_zup=False (default) to keep COLMAP convention.")
    
    return output_path


def save_trajectory_comparison_image(
    data_dir: Path,
    output_path: Path,
    traj_before: Optional[np.ndarray] = None,
    traj_after: Optional[np.ndarray] = None,
    up_offset_meters: float = 0.1,
    traj_type: Literal["interp", "ellipse", "spiral", "wall"] = "interp",
    width: int = 1920,
    height: int = 1080,
    wall_height: Optional[float] = None,
    wall_pitch_degrees: float = 15.0,
    wall_distance: float = 1.2,
    wall_shape: Literal["rectangle", "ellipse"] = "ellipse",
    n_frames: Optional[int] = None,
):
    """
    Save a rendered image of the trajectory comparison (non-interactive).
    
    Args:
        data_dir: Path to COLMAP dataset directory
        output_path: Path to save the output image
        traj_before: Original trajectory [N, 4, 4] (blue)
        traj_after: Offset trajectory [N, 4, 4] (red)
        up_offset_meters: Offset to apply if traj_after not provided
        traj_type: Trajectory type to generate
        width: Image width
        height: Image height
        wall_height: Camera height for wall trajectory
        wall_pitch_degrees: Downward pitch for wall trajectory
        wall_distance: How close to walls for wall trajectory
        wall_shape: Path shape for wall trajectory
        n_frames: Number of frames in trajectory
    """
    import open3d as o3d
    
    # Load data
    points, colors, camtoworlds_original, scene_scale = colmap_parser(data_dir)
    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=8)
    
    if traj_before is None:
        traj_before = generate_trajectory_from_parser(
            parser, traj_type="interp", scene_scale=scene_scale,
            n_frames=n_frames,
        )
    
    if traj_after is None:
        traj_after = generate_trajectory_from_parser(
            parser, 
            traj_type=traj_type, 
            scene_scale=scene_scale,
            n_frames=n_frames,
            elevation=wall_height,
            pitch_degrees=wall_pitch_degrees,
            wall_distance=wall_distance,
            wall_shape=wall_shape,
        )
    
    # Create geometries
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    subsample = max(1, len(traj_before) // 200)
    
    lineset_before = create_trajectory_lineset(traj_before, color=[0.0, 0.0, 1.0], subsample=subsample)
    lineset_after = create_trajectory_lineset(traj_after, color=[1.0, 0.0, 0.0], subsample=subsample)
    
    # Setup visualizer for offscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    vis.add_geometry(pcd)
    vis.add_geometry(lineset_before)
    vis.add_geometry(lineset_after)
    
    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    
    # Render and save
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path))
    vis.destroy_window()
    
    print(f"Saved trajectory comparison to {output_path}")


if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Visualize camera trajectories on point cloud")
    arg_parser.add_argument("--data-dir", type=str, required=True, help="Path to COLMAP dataset")
    arg_parser.add_argument("--up-offset", type=float, default=0.1, help="Up offset in meters")
    arg_parser.add_argument("--traj-type", type=str, default="interp", 
                           choices=["interp", "ellipse", "spiral", "wall", "elevated", "colmap"])
    arg_parser.add_argument("--save-image", type=str, default=None, 
                           help="Save image instead of interactive view")
    
    # Interactive picking mode
    arg_parser.add_argument("--pick", action="store_true",
                           help="Interactive mode: click to select camera positions for trajectory")
    arg_parser.add_argument("--pick-output", type=str, default=None,
                           help="Save picked trajectory to this .npy file")
    arg_parser.add_argument("--pick-closed-loop", action="store_true",
                           help="Connect last picked point back to first")
    arg_parser.add_argument("--pick-look-along-path", action="store_true",
                           help="Cameras face along path instead of toward center")
    
    # Wall/elevated trajectory options
    arg_parser.add_argument("--wall-height", type=float, default=None,
                           help="Camera height (Z) for wall/elevated/pick trajectory. None uses mean.")
    arg_parser.add_argument("--wall-pitch", type=float, default=15.0,
                           help="Downward pitch in degrees (positive = look down)")
    arg_parser.add_argument("--wall-distance", type=float, default=1.2,
                           help="Distance multiplier (1.0=at camera bounds, >1.0=outside toward walls)")
    arg_parser.add_argument("--wall-shape", type=str, default="ellipse",
                           choices=["ellipse", "rectangle"],
                           help="Shape of the wall trajectory path")
    arg_parser.add_argument("--n-frames", type=int, default=None,
                           help="Number of frames in trajectory")
    arg_parser.add_argument("--n-interp", type=int, default=5,
                           help="Interpolation points between keyframes (for pick mode)")
    arg_parser.add_argument("--save-baseline", action="store_true",
                           help="Save baseline.npy with original COLMAP camera poses to data directory")
    arg_parser.add_argument("--baseline-output", type=str, default=None,
                           help="Path to save baseline.npy (default: data_dir/baseline.npy)")
    
    args = arg_parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.save_baseline:
        # Save baseline trajectory
        save_baseline_trajectory(
            data_dir,
            output_path=Path(args.baseline_output) if args.baseline_output else data_dir / "baseline.npy",
            factor=1,
        )
    
    # if args.pick:
    #     # Interactive picking mode
    #     interactive_pick_trajectory(
    #         data_dir,
    #         height=args.wall_height,
    #         pitch_degrees=args.wall_pitch,
    #         n_interp=args.n_interp,
    #         look_at_center=not args.pick_look_along_path,
    #         closed_loop=args.pick_closed_loop,
    #         output_path=Path(args.pick_output) if args.pick_output else None,
    #     )
    # elif args.save_image:
    #     save_trajectory_comparison_image(
    #         data_dir,
    #         output_path=Path(args.save_image),
    #         up_offset_meters=args.up_offset,
    #         traj_type=args.traj_type,
    #         wall_height=args.wall_height,
    #         wall_pitch_degrees=args.wall_pitch,
    #         wall_distance=args.wall_distance,
    #         wall_shape=args.wall_shape,
    #         n_frames=args.n_frames,
    #     )
    # else:
    #     view_trajectory(
    #         data_dir,
    #         up_offset_meters=args.up_offset,
    #         traj_type=args.traj_type,
    #         wall_height=args.wall_height,
    #         wall_pitch_degrees=args.wall_pitch,
    #         wall_distance=args.wall_distance,
    #         wall_shape=args.wall_shape,
    #         n_frames=args.n_frames,
    #     )
