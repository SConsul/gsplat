"""
Trajectory utilities for camera path generation and manipulation.
"""

from pathlib import Path
from typing import Optional, Literal
from colmap_loader import colmap_parser, load_pcd_from_dir
from viz_utils import create_camera_frustum, create_trajectory_lineset, create_trajectory_spheres
from trajectory_data import TrajectoryData
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
                y[i] = (
                    center[1]
                    - wall_distance * extent_y
                    + 2 * wall_distance * extent_y * side_pos
                )
            elif side_pos < 2:  # Top side (going left in X)
                x[i] = (
                    center[0]
                    + wall_distance * extent_x
                    - 2 * wall_distance * extent_x * (side_pos - 1)
                )
                y[i] = center[1] + wall_distance * extent_y
            elif side_pos < 3:  # Left side (going down in Y)
                x[i] = center[0] - wall_distance * extent_x
                y[i] = (
                    center[1]
                    + wall_distance * extent_y
                    - 2 * wall_distance * extent_y * (side_pos - 2)
                )
            else:  # Bottom side (going right in X)
                x[i] = (
                    center[0]
                    - wall_distance * extent_x
                    + 2 * wall_distance * extent_x * (side_pos - 3)
                )
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
        right = np.array([-forward[1], forward[0], 0.0])
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
        t_interp = np.linspace(
            0, n_cameras - 1, n_cameras + (n_cameras - 1) * (n_interp - 1)
        )

        interp_func = interp1d(t_original, elevated_positions, axis=0, kind="cubic")
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
        right = np.array([-forward[1], forward[0], 0.0])
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
    center_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a trajectory from user-selected points at a fixed height with pitch.

    Args:
        selected_positions: Selected XY(Z) positions [N, 2] or [N, 3]
        height: Camera Z height. If None, uses mean Z of selected points (or 0 if 2D).
        pitch_degrees: Downward pitch angle in degrees (positive = looking down)
        n_interp: Number of interpolation points between each selected point
        look_at_center: If True, cameras face toward center. If False, face along path.
        closed_loop: If True, connect last point back to first.
        center_override: If provided, use this XY position [2,] as center instead of centroid

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
    if center_override is not None:
        center = np.array(center_override[:2])  # Use override (XY only)
    else:
        center = positions[:, :2].mean(axis=0)  # Use centroid

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

        if closed_loop and n_key >= 3:
            # For periodic spline, append first point to end (required by scipy)
            positions_periodic = np.vstack([positions, positions[0:1]])
            n_key_periodic = len(positions_periodic)
            t_original = np.linspace(0.0, 1.0, n_key_periodic)
            n_total = n_key + (n_key - 1) * (n_interp - 1)
            t_interp = np.linspace(0.0, 1.0, n_total)

            # Use a periodic cubic spline for smooth closed loops
            spline = CubicSpline(
                t_original, positions_periodic, axis=0, bc_type="periodic"
            )
            positions = spline(t_interp)
            # Add first interpolated point to end to properly close the loop
            positions = np.vstack([positions, positions[0:1]])
        else:
            # Standard cubic interpolation for open paths
            t_original = np.linspace(0.0, 1.0, n_key)
            n_total = n_key + (n_key - 1) * (n_interp - 1)
            t_interp = np.linspace(0.0, 1.0, n_total)
            interp_func = interp1d(t_original, positions, axis=0, kind="cubic")
            positions = interp_func(t_interp)

    n_frames = len(positions)
    print(f"  Total frames: {n_frames}")
    if closed_loop:
        print(f"  Loop closed: first and last positions are identical")

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

        right = np.array([-forward[1], forward[0], 0.0])
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
    pick_center: bool = False,
    output_path: Optional[Path] = None,
) -> Optional[TrajectoryData]:
    """
    Interactive mode to pick waypoints (XY only) and optionally a center point.

    Opens a visualization where you can:
    1. Shift+Click on points to select waypoints (only XY used, Z set by height)
    2. If pick_center=True, first click picks the center/focus point
    3. Select points in the ORDER you want the trajectory to follow
    4. Press 'Q' to finish selection and generate trajectory
    5. View the generated trajectory

    Args:
        data_dir: Path to COLMAP dataset directory
        height: Camera Z height for all waypoints. None uses mean Z of scene.
        pitch_degrees: Downward pitch angle in degrees (positive = looking down)
        n_interp: Interpolation points between selected points
        look_at_center: If True, cameras face toward center. If False, follow loop.
        closed_loop: If True, connect last point back to first
        pick_center: If True, first click picks the center/focus point
        output_path: If provided, save trajectory to this file (.npz format)

    Returns:
        TrajectoryData instance with generated trajectory, or None if cancelled
    """
    import open3d as o3d

    print(f"Loading COLMAP data from {data_dir}...")
    colmap = load_pcd_from_dir(data_dir)
    
    pcd = colmap.viz_o3d(show_cam=True)

    # Determine default height if not provided
    if height is None:
        height = colmap.points[:, 2].mean()
        print(f"\nUsing auto height: {height:.3f}")

    center_point_xy = None
    waypoint_positions_xy = []

    # Calculate scene-relative sphere sizes
    scene_extent = colmap.extent
    center_sphere_radius = max(scene_extent) * 0.015  # 1.5% for center
    waypoint_sphere_radius = max(scene_extent) * 0.01  # 1% for waypoints

    # If pick_center is enabled, pick center point first in a separate session
    if pick_center and look_at_center:
        print(f"\n{'=' * 60}")
        print("STEP 1: PICK CENTER POINT")
        print(f"{'=' * 60}")
        print("Instructions:")
        print("  1. [SHIFT + LEFT CLICK] to select CENTER/FOCUS point")
        print("  2. [SHIFT + RIGHT CLICK] to undo point picking")
        print("  3. Can pick multiple times - LAST click will be used")
        print("  4. Press 'Q' to close window and move to waypoint selection")
        print(f"\nTips:")
        print(f"  - GREEN points = original camera positions")
        print(f"  - Only XY coordinate is used; Z is set by height parameter")
        print(f"  - Picked point will be shown with CYAN sphere in next view")
        print(f"{'=' * 60}\n")

        vis_center = o3d.visualization.VisualizerWithEditing()
        vis_center.create_window(
            window_name="Pick Center Point (Shift+Click, Q when done)",
            width=1280,
            height=720,
        )
        vis_center.add_geometry(pcd)
        vis_center.run()
        picked_center = vis_center.get_picked_points()
        vis_center.destroy_window()

        if len(picked_center) > 0:
            # Use the last picked point as center
            center_idx = picked_center[-1]
            center_pos = combined_points[center_idx]
            center_point_xy = center_pos[:2]
            point_type = "camera" if center_idx >= n_scene_points else "scene"
            if len(picked_center) > 1:
                print(f"\nPicked {len(picked_center)} points, using LAST one as center:")
            print(f"  CENTER: [{point_type}] XY=({center_pos[0]:.3f}, {center_pos[1]:.3f})")

            # Show preview with ONLY the latest center point
            print("\n" + "="*60)
            print("PREVIEW: Selected Center (Latest Only)")
            print("="*60)
            print("  - CYAN sphere = your selected center (latest pick only)")
            print("Close window to continue...")
            print("="*60 + "\n")

            vis_preview = o3d.visualization.Visualizer()
            vis_preview.create_window(window_name="Preview: Center", width=1280, height=720)
            vis_preview.add_geometry(pcd)

            # Render ONLY the latest center point with large cyan sphere
            center_3d = np.array([center_point_xy[0], center_point_xy[1], height])
            center_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=center_sphere_radius, resolution=20
            )
            center_sphere.translate(center_3d)
            center_sphere.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
            center_sphere.compute_vertex_normals()
            vis_preview.add_geometry(center_sphere)

            vis_preview.run()
            vis_preview.destroy_window()
        else:
            print("\nNo center point selected, will use waypoint centroid as center")

    # Pick waypoints
    print(f"\n{'=' * 60}")
    print(
        "STEP {}: PICK WAYPOINTS".format(2 if (pick_center and look_at_center) else 1)
    )
    print(f"{'=' * 60}")
    print("Instructions:")
    print("  1. [SHIFT + LEFT CLICK] to select waypoints (XY only)")
    print("  2. [SHIFT + RIGHT CLICK] to undo point picking")
    print("  3. Click points in the ORDER you want the trajectory to follow")
    print("  4. Press 'Q' to close window and generate trajectory")
    print(f"\nSettings:")
    print(f"  - Mode: {'Look at center' if look_at_center else 'Follow loop'}")
    print(f"  - Height: {height:.3f}")
    print(f"  - Pitch: {pitch_degrees}°")
    print(f"  - Closed loop: {closed_loop}")
    print(f"\nTips:")
    print(f"  - GREEN points = original camera positions")
    if center_point_xy is not None:
        print(f"  - Center point (CYAN sphere) will be shown in preview")
    print(f"  - Only XY coordinates are used from clicks; Z is set by height={height:.3f}")
    print(f"  - Selected waypoints will be shown with YELLOW spheres in preview")
    print(f"{'=' * 60}\n")

    vis_waypoints = o3d.visualization.VisualizerWithEditing()
    vis_waypoints.create_window(
        window_name="Pick Waypoints (Shift+Click, Q when done)", width=1280, height=720
    )
    vis_waypoints.add_geometry(pcd)
    vis_waypoints.run()
    picked_waypoints = vis_waypoints.get_picked_points()
    vis_waypoints.destroy_window()

    if len(picked_waypoints) < 2:
        print(
            f"ERROR: Only {len(picked_waypoints)} waypoints selected. Need at least 2."
        )
        return None

    # Extract waypoint positions (XY only)
    print(f"\nSelected {len(picked_waypoints)} waypoint(s):")
    try:
        for i, idx in enumerate(picked_waypoints, start=1):
            if idx >= len(combined_points):
                print(f"ERROR: Index {idx} out of bounds (max {len(combined_points)-1})")
                return None
            pos = combined_points[idx]
            waypoint_positions_xy.append(pos[:2])  # Only XY
            point_type = "camera" if idx >= n_scene_points else "scene"
            print(
                f"  {i}. [{point_type}] XY=({pos[0]:.3f}, {pos[1]:.3f}) -> will use Z={height:.3f}"
            )

        waypoint_positions_xy = np.array(waypoint_positions_xy)
    except Exception as e:
        print(f"\nERROR processing waypoints: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Show preview with picked points as large spheres
    print("\n" + "="*60)
    print("PREVIEW: Picked Points")
    print("="*60)
    print("Showing your selections with large spheres:")
    if center_point_xy is not None:
        print("  - CYAN sphere = center point")
    print("  - YELLOW spheres = waypoints")
    print("  - GREEN points = original cameras")
    print("Close window to continue...")
    print("="*60 + "\n")

    try:
        vis_preview = o3d.visualization.Visualizer()
        vis_preview.create_window(window_name="Preview: Picked Points", width=1280, height=720)
        vis_preview.add_geometry(pcd)

        # Show center point with large cyan sphere
        if center_point_xy is not None:
            center_3d = np.array([center_point_xy[0], center_point_xy[1], height])
            center_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=center_sphere_radius, resolution=20
            )
            center_sphere.translate(center_3d)
            center_sphere.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
            center_sphere.compute_vertex_normals()
            vis_preview.add_geometry(center_sphere)

        # Show waypoints with large yellow spheres
        for wp_xy in waypoint_positions_xy:
            wp_3d = np.array([wp_xy[0], wp_xy[1], height])
            wp_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=waypoint_sphere_radius, resolution=20
            )
            wp_sphere.translate(wp_3d)
            wp_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
            wp_sphere.compute_vertex_normals()
            vis_preview.add_geometry(wp_sphere)

        vis_preview.run()
        vis_preview.destroy_window()
    except Exception as e:
        print(f"\nERROR creating preview window: {e}")
        import traceback
        traceback.print_exc()
        # Continue anyway - preview is optional

    # Build 3D waypoints with specified height
    waypoint_positions_3d = np.column_stack(
        [waypoint_positions_xy, np.full(len(waypoint_positions_xy), height)]
    )

    # Determine center for look_at
    if look_at_center:
        if center_point_xy is not None:
            # Use picked center
            center_for_lookat = center_point_xy
            print(
                f"\nUsing picked center: ({center_for_lookat[0]:.3f}, {center_for_lookat[1]:.3f})"
            )
        else:
            # Use centroid of waypoints
            center_for_lookat = waypoint_positions_xy.mean(axis=0)
            print(
                f"\nUsing waypoint centroid as center: ({center_for_lookat[0]:.3f}, {center_for_lookat[1]:.3f})"
            )
    else:
        center_for_lookat = None

    # Generate trajectory from selected points
    trajectory = generate_trajectory_from_selected_points(
        waypoint_positions_3d,
        height=height,
        pitch_degrees=pitch_degrees,
        n_interp=n_interp,
        look_at_center=look_at_center,
        closed_loop=closed_loop,
        center_override=center_for_lookat if center_for_lookat is not None else None,
    )

    # Get camera intrinsics for TrajectoryData
    parser = Parser(
        data_dir=str(data_dir),
        factor=1,
        normalize=True,
        test_every=8,
    )
    first_camera_id = list(parser.Ks_dict.keys())[0]
    K = parser.Ks_dict[first_camera_id]
    img_width, img_height = parser.imsize_dict[first_camera_id]

    # Create TrajectoryData
    traj_data = TrajectoryData(
        camtoworlds=trajectory,
        intrinsics=K,
        width=img_width,
        height=img_height,
        waypoints=waypoint_positions_xy,
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

    # Save trajectory
    if output_path is not None:
        if trajectory is not None:
            traj_data.save(output_path)
            print(
                f"  python render_trajectory.py --data-dir {data_dir} --ckpt <checkpoint.pt> \\"
            )
            print(f"      --traj-file {output_path} -o output.mp4")

    # Visualize the result
    print("\nShowing generated trajectory...")

    # Calculate scene-relative sphere sizes for final visualization
    final_camera_radius = max(scene_extent) * 0.005  # 0.5% for original camera positions
    final_waypoint_radius = max(scene_extent) * 0.01  # 1% for waypoints
    final_center_radius = max(scene_extent) * 0.015  # 1.5% for center

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Generated Trajectory", width=1280, height=720)

    vis2.add_geometry(pcd)

    # Show original cameras in green (small)
    for pos in cam_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=final_camera_radius)
        sphere.translate(pos)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        vis2.add_geometry(sphere)

    # Show center point in cyan (if picked) - largest
    if center_point_xy is not None:
        center_3d = np.array([center_point_xy[0], center_point_xy[1], height])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=final_center_radius)
        sphere.translate(center_3d)
        sphere.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
        vis2.add_geometry(sphere)

    # Show selected waypoints in yellow (large and visible)
    for pos_xy in waypoint_positions_xy:
        pos_3d = np.array([pos_xy[0], pos_xy[1], height])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=final_waypoint_radius)
        sphere.translate(pos_3d)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        vis2.add_geometry(sphere)

    # Show generated trajectory in red
    if trajectory is not None:
        traj_positions = trajectory[:, :3, 3]

        # Draw trajectory path as red line
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(traj_positions)
        lines = [[i, i + 1] for i in range(len(traj_positions) - 1)]
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(lines))
        vis2.add_geometry(lineset)

        # Helper function to create camera frustum
        

        # Show camera frustums at intervals along trajectory
        frustum_size = max(scene_extent) * 0.05  # 5% of scene extent
        skip = max(1, len(trajectory) // 20)  # Show ~20 frustums

        for i in range(0, len(trajectory), skip):
            frustum = create_camera_frustum(
                trajectory[i],
                size=frustum_size,
                color=[1.0, 0.5, 0.0]  # Orange
            )
            vis2.add_geometry(frustum)

        # Show trajectory camera positions as small spheres
        for i, pos in enumerate(traj_positions[::skip]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=final_camera_radius * 0.5)
            sphere.translate(pos)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            vis2.add_geometry(sphere)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis2.add_geometry(coord_frame)

    print("\nVisualization legend:")
    print("  GREEN spheres  = Original camera positions")
    if center_point_xy is not None:
        print("  CYAN sphere    = Center/focus point")
    print("  YELLOW spheres = Selected waypoints")
    print("  RED line       = Generated trajectory path")
    print("  RED spheres    = Trajectory camera positions")
    print("  ORANGE frustums = Camera view frustums (orientation)")
    print("\nNote: Frustums show camera orientation - pointing direction and field of view")

    vis2.run()
    vis2.destroy_window()

    return traj_data


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
    transform = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

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


def offset_trajectory(camtoworlds: np.ndarray, offset: np.ndarray) -> np.ndarray:
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
            K = np.array(
                [
                    [0, -right[2], right[1]],
                    [right[2], 0, -right[0]],
                    [-right[1], right[0], 0],
                ]
            )

            # Rodrigues rotation formula
            pitch_rotation = np.eye(3) + sin_p * K + (1 - cos_p) * (K @ K)

            # Apply pitch rotation to the camera orientation
            # New orientation = pitch_rotation @ old_orientation
            result[i, :3, :3] = pitch_rotation @ R

        print(
            f"Applied pitch rotation: {pitch_degrees:.1f}° around each camera's right axis"
        )

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
            np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0
        )
        camtoworlds = np.concatenate([camtoworlds, homogeneous], axis=1)
    return camtoworlds


def generate_trajectory_from_parser(
    parser: Parser,
    traj_type: Literal[
        "interp", "ellipse", "spiral", "wall", "elevated", "colmap"
    ] = "interp",
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
        camtoworlds_all = apply_elevation_and_pitch(
            camtoworlds_all, elevation=elevation, pitch_degrees=pitch_degrees
        )
    elif traj_type == "ellipse":
        # Elliptical path at average height
        height = camtoworlds_all[:, 2, 3].mean()
        n = 120 if n_frames is None else n_frames
        camtoworlds_all = generate_ellipse_path_z(
            camtoworlds_all, n_frames=n, height=height
        )
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
        n_interp = (
            1
            if n_frames is None
            else max(1, n_frames // (len(camtoworlds_all) // elevated_subsample))
        )
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
    traj_type: Literal[
        "interp", "ellipse", "spiral", "wall", "elevated", "colmap"
    ] = "interp",
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
    scaled_offset = (
        up_offset_meters / scene_scale if scene_scale != 0 else up_offset_meters
    )
    offset = np.array([0.0, 0.0, -scaled_offset])  # -Z is up

    camtoworlds = offset_trajectory(camtoworlds, offset)

    return camtoworlds




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
    pcd, scene_scale = load_pcd_from_dir(data_dir)

    geometries = [pcd]

    # Generate trajectories if not provided
    parser = Parser(str(data_dir), factor=1, normalize=True, test_every=8)

    if traj_before is None:
        print(f"[BEFORE] Generating interpolated original camera trajectory...")
        traj_before = generate_trajectory_from_parser(
            parser,
            traj_type="interp",
            scene_scale=scene_scale,
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
    lineset_before = create_trajectory_lineset(
        traj_before, color=[0.0, 0.0, 1.0], subsample=subsample
    )
    geometries.append(lineset_before)

    spheres_before = create_trajectory_spheres(
        traj_before, color=[0.0, 0.0, 1.0], radius=0.015, subsample=subsample * 5
    )
    geometries.extend(spheres_before)

    # After trajectory: RED lines and spheres
    lineset_after = create_trajectory_lineset(
        traj_after, color=[1.0, 0.0, 0.0], subsample=subsample
    )
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
    print(
        "\nControls: Left-click drag to rotate, scroll to zoom, right-click drag to pan"
    )

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
    print(
        f"Saved baseline camera trajectory ({len(parser.camtoworlds)} poses) to {output_path}"
    )
    print("  Note: baseline.npy is in COLMAP convention (Y-down). When loaded with")
    print(
        "  load_trajectory(), use convert_zup=False (default) to keep COLMAP convention."
    )

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
            parser,
            traj_type="interp",
            scene_scale=scene_scale,
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

    lineset_before = create_trajectory_lineset(
        traj_before, color=[0.0, 0.0, 1.0], subsample=subsample
    )
    lineset_after = create_trajectory_lineset(
        traj_after, color=[1.0, 0.0, 0.0], subsample=subsample
    )

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

    arg_parser = argparse.ArgumentParser(
        description="Visualize camera trajectories on point cloud"
    )
    arg_parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to COLMAP dataset"
    )
    arg_parser.add_argument(
        "--up-offset", type=float, default=0.1, help="Up offset in meters"
    )
    arg_parser.add_argument(
        "--traj-type",
        type=str,
        default="interp",
        choices=["interp", "ellipse", "spiral", "wall", "elevated", "colmap"],
    )
    arg_parser.add_argument(
        "--save-image",
        type=str,
        default=None,
        help="Save image instead of interactive view",
    )

    # Interactive picking mode
    arg_parser.add_argument(
        "--pick",
        action="store_true",
        help="Interactive mode: click to select camera positions for trajectory",
    )
    arg_parser.add_argument(
        "--pick-output",
        type=str,
        default=None,
        help="Save picked trajectory to this .npz file (TrajectoryData format)",
    )
    arg_parser.add_argument(
        "--pick-closed-loop",
        action="store_true",
        help="Connect last picked point back to first",
    )
    arg_parser.add_argument(
        "--pick-look-along-path",
        action="store_true",
        help="Cameras face along path instead of toward center",
    )
    arg_parser.add_argument(
        "--pick-center",
        action="store_true",
        help="Two-stage picking: (1) pick center (can click multiple, last kept), (2) pick waypoints",
    )

    # Wall/elevated trajectory options
    arg_parser.add_argument(
        "--wall-height",
        type=float,
        default=None,
        help="Camera height (Z) for wall/elevated/pick trajectory. None uses mean.",
    )
    arg_parser.add_argument(
        "--wall-pitch",
        type=float,
        default=15.0,
        help="Downward pitch in degrees (positive = look down)",
    )
    arg_parser.add_argument(
        "--wall-distance",
        type=float,
        default=1.2,
        help="Distance multiplier (1.0=at camera bounds, >1.0=outside toward walls)",
    )
    arg_parser.add_argument(
        "--wall-shape",
        type=str,
        default="ellipse",
        choices=["ellipse", "rectangle"],
        help="Shape of the wall trajectory path",
    )
    arg_parser.add_argument(
        "--n-frames", type=int, default=None, help="Number of frames in trajectory"
    )
    arg_parser.add_argument(
        "--n-interp",
        type=int,
        default=5,
        help="Interpolation points between keyframes (for pick mode)",
    )
    arg_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save baseline.npy with original COLMAP camera poses to data directory",
    )
    arg_parser.add_argument(
        "--baseline-output",
        type=str,
        default=None,
        help="Path to save baseline.npy (default: data_dir/baseline.npy)",
    )

    args = arg_parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.save_baseline:
        # Save baseline trajectory
        save_baseline_trajectory(
            data_dir,
            output_path=Path(args.baseline_output)
            if args.baseline_output
            else data_dir / "baseline.npy",
            factor=1,
        )

    if args.pick:
        # Interactive picking mode
        interactive_pick_trajectory(
            data_dir,
            height=args.wall_height,
            pitch_degrees=args.wall_pitch,
            n_interp=args.n_interp,
            look_at_center=not args.pick_look_along_path,
            closed_loop=args.pick_closed_loop,
            pick_center=args.pick_center,
            output_path=Path(args.pick_output) if args.pick_output else None,
        )
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
