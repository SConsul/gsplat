from colmap_loader import MyColmap

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajectory_data import TrajectoryData


def view_pcd_with_traj(
    pcd: MyColmap,
    trajs: list["TrajectoryData"] | None,
    frustum_skip: int = 20,
    sphere_skip: int = 10,
):
    import open3d as o3d
    # Calculate scene extent for scaling
    scene_extent = pcd.extent
    max_extent = max(scene_extent)

    frustum_size = max(scene_extent) * 0.001
    sphere_radius = max(scene_extent) * 0.00015

    # Adjust visualization sizes based on scene scale
    adjusted_frustum_size = frustum_size * max_extent
    adjusted_sphere_radius = sphere_radius * max_extent

    print(f"Scene extent: {max_extent:.3f}")
    print(f"Adjusted frustum size: {adjusted_frustum_size:.3f}")
    print(f"Adjusted sphere radius: {adjusted_sphere_radius:.3f}")

    # Create visualization geometries
    geometries = [pcd.viz_o3d()]

    if trajs is not None:
        for traj in trajs:
            # Add trajectory line (red)
            print("\nCreating trajectory visualization...")
            lineset = create_trajectory_lineset(
                traj.camtoworlds, color=[1.0, 0.0, 0.0], subsample=1
            )
            geometries.append(lineset)

            # Add camera position spheres (red, smaller)
            spheres = create_trajectory_spheres(
                traj.camtoworlds,
                color=[1.0, 0.0, 0.0],
                radius=adjusted_sphere_radius * 0.5,
                subsample=sphere_skip,
            )
            geometries.extend(spheres)

            # Add camera frustums (orange)
            print(f"Adding camera frustums (every {frustum_skip} frames)...")
            for i in range(0, len(traj.camtoworlds), frustum_skip):
                frustum = create_camera_frustum(
                    traj.camtoworlds[i],
                    size=adjusted_frustum_size,
                    color=[1.0, 0.5, 0.0],  # Orange
                )
                geometries.append(frustum)

            # Show waypoints if available (yellow spheres, larger)
            if traj.waypoints is not None:
                print(f"Adding {len(traj.waypoints)} waypoints...")
                for wp in traj.waypoints:
                    # Waypoints are typically XY, need to add Z
                    if len(wp) == 2:
                        # Use mean height from trajectory
                        z = traj.camtoworlds[:, 2, 3].mean()
                        wp_3d = np.array([wp[0], wp[1], z])
                    else:
                        wp_3d = wp

                    sphere = o3d.geometry.TriangleMesh.create_sphere(
                        radius=adjusted_sphere_radius * 2.0
                    )
                    sphere.translate(wp_3d)
                    sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
                    sphere.compute_vertex_normals()
                    geometries.append(sphere)

            # Show center point if available (cyan sphere, largest)
            if traj.center_point is not None:
                print("Adding center point...")
                if len(traj.center_point) == 2:
                    z = traj.camtoworlds[:, 2, 3].mean()
                    center_3d = np.array(
                        [traj.center_point[0], traj.center_point[1], z]
                    )
                else:
                    center_3d = traj.center_point

                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=adjusted_sphere_radius * 3.0
                )
                sphere.translate(center_3d)
                sphere.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
                sphere.compute_vertex_normals()
                geometries.append(sphere)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max_extent * 0.2
    )
    geometries.append(coord_frame)

    # Display
    print("\nVisualization legend:")
    print("  RED line       = Trajectory path")
    print("  RED spheres    = Camera positions")
    print("  ORANGE frustums = Camera orientations")
    if trajs is not None:
        if trajs[0].waypoints is not None:
            print("  YELLOW spheres = Waypoints")
        if trajs[0].center_point is not None:
            print("  CYAN sphere    = Center point")
    print("\nClose window to exit...")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Trajectory Viewer",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def create_trajectory_spheres(
    camtoworlds: np.ndarray,
    color: list[float] = [0.0, 0.0, 1.0],
    radius: float = 0.02,
    subsample: int = 10,
) -> list["o3d.geometry.TriangleMesh"]:
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
    positions = camtoworlds[::subsample, :3, 3]
    spheres = []

    for pos in positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    return spheres


def create_camera_frustum(
    pose, size=0.1, color=[1.0, 0.0, 0.0]
) -> "o3d.geometry.LineSet":
    """Create a camera frustum wireframe."""
    # Camera center
    center = pose[:3, 3]

    # Camera axes
    x_axis = pose[:3, 0]  # Right
    y_axis = pose[:3, 1]  # Down (or up depending on convention)
    z_axis = pose[:3, 2]  # Forward (camera looking direction)

    # Define frustum corners in camera space (simple pyramid)
    # Image plane at distance 'size' from camera center
    aspect = 1.0
    half_width = size * aspect * 0.5
    half_height = size * 0.5
    depth = size

    # Four corners of image plane
    corners = [
        center
        + depth * z_axis
        + half_width * x_axis
        - half_height * y_axis,  # top-right
        center
        + depth * z_axis
        - half_width * x_axis
        - half_height * y_axis,  # top-left
        center
        + depth * z_axis
        - half_width * x_axis
        + half_height * y_axis,  # bottom-left
        center
        + depth * z_axis
        + half_width * x_axis
        + half_height * y_axis,  # bottom-right
    ]

    # Create lineset for frustum
    points = [center] + corners
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],  # Lines from center to corners
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],  # Rectangle at image plane
    ]

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return frustum


def create_trajectory_lineset(
    camtoworlds: np.ndarray,
    color: list[float] = [0.0, 0.0, 1.0],
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
