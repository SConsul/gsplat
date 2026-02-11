from pathlib import Path
from datasets.colmap import Parser
from dataclasses import dataclass
import numpy as np


@dataclass
class MyColmap:
    points: np.ndarray
    point_rgb: np.ndarray
    cam_positions: np.ndarray
    
    @property
    def extent(self):
        return self.points.max(axis=0) - self.points.min(axis=0)

    @property
    def center(self):
        return self.points.max(axis=0) - self.points.min(axis=0)
    
    def viz_o3d(self, show_cam: bool=False) -> "o3d.geometry.PointCloud":
        import open3d as o3d

        if show_cam:
            combined_points = np.vstack([self.points, self.cam_positions])
            cam_colors = np.array([[0.0, 1.0, 0.0]] * len(self.cam_positions))  # Green for cameras
            combined_colors = np.vstack([self.point_rgb, cam_colors])
        else:
            combined_points = self.points
            combined_colors = self.point_rgb
            
        n_axis_points = 20  # Number of points per axis
        axis_points_list = []
        axis_colors_list = []

        # X axis (red) - from center along +X
        frame_size = max(self.extent) * 0.15  # 15% of scene extent
        for i in range(n_axis_points + 1):
            t = i / n_axis_points
            axis_points_list.append(self.center + [t * frame_size, 0, 0])
            axis_colors_list.append([1.0, 0.0, 0.0])  # Red

        # Y axis (green) - from center along +Y
        for i in range(n_axis_points + 1):
            t = i / n_axis_points
            axis_points_list.append(self.center + [0, t * frame_size, 0])
            axis_colors_list.append([0.0, 1.0, 0.0])  # Green

        # Z axis (blue) - from center along +Z
        for i in range(n_axis_points + 1):
            t = i / n_axis_points
            axis_points_list.append(self.center + [0, 0, t * frame_size])
            axis_colors_list.append([0.0, 0.0, 1.0])  # Blue

        # Add axis points to combined point cloud
        axis_points_array = np.array(axis_points_list)
        axis_colors_array = np.array(axis_colors_list)
        combined_points = np.vstack([combined_points, axis_points_array])
        combined_colors = np.vstack([combined_colors, axis_colors_array])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        return pcd

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
        existing_files = [
            f for f in existing_files if f.is_file() and not f.name.startswith(".")
        ]
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
        base_images = [
            f for f in base_images if f.is_file() and not f.name.startswith(".")
        ]
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


def colmap_parser(
    data_dir: Path, factor: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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


def load_pcd_from_dir(data_dir: Path, save: bool) -> MyColmap:
    points, colors, camtoworlds_original, scene_scale = colmap_parser(data_dir)

    cam_positions = camtoworlds_original[:, :3, 3]
    n_scene_points = len(points)
    n_cameras = len(cam_positions)

    print(f"Loaded {n_scene_points} scene points, {n_cameras} cameras")

    if save:
        # Dump original camera trajectory for debugging
        baseline_path = Path(data_dir) / "baseline.npy"
        np.save(baseline_path, camtoworlds_original)
        print(f"Saved baseline camera trajectory to {baseline_path}")

    # Print bounds to help user understand the scene
    print(f"\nScene bounds:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    return MyColmap(points=points, point_rgb=colors, cam_positions=cam_positions)
