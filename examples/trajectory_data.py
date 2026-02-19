from dataclasses import dataclass
import copy
from enum import StrEnum

from colmap_loader import load_pcd_from_dir
import json
from pathlib import Path
from viz_utils import view_pcd_with_traj
import numpy as np
import click

class TrajectoryType(StrEnum):
    INTERP = "interp"
    ELLIPSE = "ellipse"
    SPIRAL = "spiral"
    WALL = "wall"
    ELEVATED = "elevated"
    COLMAP = "colmap"
    
class WallTrajectoryShape(StrEnum):
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"

@dataclass
class TrajectoryData:
    """
    Shared trajectory data structure for camera paths.

    This dataclass is used to pass trajectory data between trajectory_local.py
    (trajectory generation) and generate_traj_local.py (trajectory rendering).

    Attributes:
        camtoworlds: Camera to world transforms [N, 4, 4]
        intrinsics: Camera intrinsic matrix [3, 3]
        width: Image width in pixels
        height: Image height in pixels
        waypoints: Optional waypoints used to generate trajectory [M, 2] or [M, 3]
        center_point: Optional center/focus point [2,] or [3,]
        metadata: Optional dictionary with additional info (height, pitch, etc.)
    """

    camtoworlds: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    waypoints: np.ndarray | None = None
    center_point: np.ndarray | None = None
    metadata: dict | None = None

    def save(self, path: Path) -> None:
        """
        Save trajectory data to file.

        Args:
            path: Output path (.npz file)
        """

        # Prepare data dictionary
        data = {
            "camtoworlds": self.camtoworlds,
            "intrinsics": self.intrinsics,
            "width": self.width,
            "height": self.height,
        }

        if self.waypoints is not None:
            data["waypoints"] = self.waypoints

        if self.center_point is not None:
            data["center_point"] = self.center_point

        if self.metadata is not None:
            # Save metadata as JSON string
            data["metadata_json"] = json.dumps(self.metadata)

        np.savez(path, **data)
        print(f"Saved trajectory data to {path}")
        print(f"  Poses: {len(self.camtoworlds)}")
        print(f"  Image size: {self.width}x{self.height}")
        if self.waypoints is not None:
            print(f"  Waypoints: {len(self.waypoints)}")
        if self.metadata is not None:
            print(f"  Metadata: {self.metadata}")

    @staticmethod
    def load(path: Path) -> "TrajectoryData":
        """
        Load trajectory data from file.

        Args:
            path: Input path (.npz file)

        Returns:
            TrajectoryData instance
        """
        data = np.load(path, allow_pickle=True)

        waypoints = data["waypoints"] if "waypoints" in data else None
        center_point = data["center_point"] if "center_point" in data else None

        metadata = None
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))

        traj_data = TrajectoryData(
            camtoworlds=data["camtoworlds"],
            intrinsics=data["intrinsics"],
            width=int(data["width"]),
            height=int(data["height"]),
            waypoints=waypoints,
            center_point=center_point,
            metadata=metadata,
        )

        print(f"Loaded trajectory data from {path}")
        print(f"  Poses: {len(traj_data.camtoworlds)}")
        print(f"  Image size: {traj_data.width}x{traj_data.height}")
        if traj_data.waypoints is not None:
            print(f"  Waypoints: {len(traj_data.waypoints)}")
        if traj_data.metadata is not None:
            print(f"  Metadata: {traj_data.metadata}")

        return traj_data

    def edit_height(self, z: float):
        """
        Edit trajectory to input z

        Args:
            z: Height to set to camera z-coordinates (in world units)

        Returns:
            Modified TrajectoryData instance for method chaining
        """

        # Create a deep copy to avoid modifying the original
        new_traj = copy.deepcopy(self)

        # Modify the z-component (height) of the translation vector
        # camtoworlds[:, 2, 3] is the z-coordinate of the camera position
        new_traj.camtoworlds[:, 2, 3] = z

        # Update metadata
        if new_traj.metadata is None:
            new_traj.metadata = {}
        new_traj.metadata["height"] = z

        return new_traj

    def edit_pitch(self, pitch: float):
        """
        Edit trajectory to make cameras tilt to input pitch

        Args:
            pitch: Pitch angle in degrees (positive = looking down, negative = looking up)

        Returns:
            Modified TrajectoryData instance for method chaining
        """
        if pitch is None:
            return self

        # Create a deep copy to avoid modifying the original
        new_traj = copy.deepcopy(self)

        # Convert pitch to radians
        pitch_rad = np.deg2rad(pitch)

        # Create rotation matrix for pitch (rotation around x-axis)
        # This rotates the camera's view direction up/down
        pitch_rotation = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)],
            ]
        )

        # Apply pitch rotation to each camera's rotation matrix
        for i in range(len(new_traj.camtoworlds)):
            # Extract the rotation part (top-left 3x3)
            R = new_traj.camtoworlds[i, :3, :3]
            # Apply pitch rotation: R_new = R @ pitch_rotation
            new_traj.camtoworlds[i, :3, :3] = R @ pitch_rotation

        # Update metadata
        if new_traj.metadata is None:
            new_traj.metadata = {}
        new_traj.metadata["pitch_offset"] = pitch

        return new_traj

    def scale(self, scale: float):
        """
        Bring the trajectory points closer to (or farther from) the focus point in XY by scale factor.

        Scales the XY distance between each camera position and the center point.
        Z-coordinates (height) remain unchanged.

        Args:
            scale: Scale factor for XY distance from center point
                   (< 1.0 brings closer, > 1.0 moves farther, 1.0 no change)

        Returns:
            Modified TrajectoryData instance for method chaining

        Raises:
            ValueError: If no center_point is defined in the trajectory
        """
        if scale is None or scale == 1.0:
            return self

        if self.center_point is None:
            raise ValueError(
                "Cannot scale trajectory: no center_point defined. "
                "Use a trajectory created with look_at_center=True and pick_center=True."
            )

        # Create a deep copy to avoid modifying the original
        new_traj = copy.deepcopy(self)

        assert len(self.center_point) == 2, (
            f"center should be XY got {self.center_point}"
        )

        # Get center point XY coordinates
        center_xy = self.center_point[:2]

        # Scale each camera position relative to center point in XY plane
        for i in range(len(new_traj.camtoworlds)):
            # Get current camera position
            cam_pos = new_traj.camtoworlds[i, :3, 3]

            # Calculate vector from center to camera (XY only)
            cam_xy = cam_pos[:2]
            offset_xy = cam_xy - center_xy

            # Scale the offset
            scaled_offset_xy = offset_xy * scale

            # Update camera XY position (keep Z unchanged)
            new_traj.camtoworlds[i, 0, 3] = center_xy[0] + scaled_offset_xy[0]
            new_traj.camtoworlds[i, 1, 3] = center_xy[1] + scaled_offset_xy[1]

        # Scale waypoints if present (XY only)
        if new_traj.waypoints is not None:
            for i in range(len(new_traj.waypoints)):
                wp_xy = new_traj.waypoints[i, :2]
                offset_xy = wp_xy - center_xy
                scaled_offset_xy = offset_xy * scale
                new_traj.waypoints[i, :2] = center_xy + scaled_offset_xy

        # Update metadata
        if new_traj.metadata is None:
            new_traj.metadata = {}
        new_traj.metadata["xy_scale_factor"] = scale

        return new_traj

    def fix_left_right_flip(self):
        """
        Fix left-right flip in trajectories generated with incorrect right vector computation.

        This corrects trajectories that were generated before the fix to generate_trajectory.py
        (lines 137, 262, 412) where the right vector was computed as [forward[1], -forward[0], 0]
        instead of the correct [-forward[1], forward[0], 0].

        The fix negates the first column of each camera's rotation matrix, which flips the
        camera's X-axis direction (left <-> right) while preserving Y and Z axes.

        Returns:
            Modified TrajectoryData instance for method chaining

        Note:
            This function should only be applied to trajectories that exhibit left-right flip.
            Applying it twice will cancel out the correction.
        """
        # Create a deep copy to avoid modifying the original
        new_traj = copy.deepcopy(self)

        # Fix each camera's rotation matrix
        for i in range(len(new_traj.camtoworlds)):
            # Extract rotation matrix (top-left 3x3)
            R = new_traj.camtoworlds[i, :3, :3].copy()

            # Negate the first column (camera's X-axis / right direction)
            # This flips left <-> right in the rendered view
            R[:, 0] = -R[:, 0]

            # Update the rotation matrix
            new_traj.camtoworlds[i, :3, :3] = R

        # Update metadata to track that this correction was applied
        if new_traj.metadata is None:
            new_traj.metadata = {}
        new_traj.metadata["left_right_flip_corrected"] = True

        print("Applied left-right flip correction to trajectory")
        print("  Negated first column of rotation matrices (camera X-axis)")

        return new_traj


@click.command()
@click.argument("input-path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("-o", "--output-path", type=click.Path(path_type=Path, exists=False))
@click.option("-h", "--height", type=float, help="Height offset to add (world units)")
@click.option(
    "-p", "--pitch", type=float, help="Pitch angle in degrees (positive = look down)"
)
@click.option(
    "-s",
    "--scale",
    type=float,
    help="Scale XY distance from center point (requires center_point)",
)
@click.option(
    "-pcd", "--pcd-path", type=click.Path(path_type=Path, exists=True, dir_okay=True)
)
def edit_traj(
    input_path: Path,
    output_path: Path,
    height: float,
    pitch: float,
    scale: float,
    pcd_path: Path | None,
):
    """
    Edit trajectory by adjusting height, pitch, and/or scale.

    Can chain multiple edits: height adjustment, pitch rotation, and XY scaling
    relative to the center point.
    """
    traj = TrajectoryData.load(input_path)
    print("=== Current Trajectory Stats")
    print(f"height={traj.camtoworlds[:5, 2, 3]}")
    new_traj = traj

    if height is not None:
        new_traj = new_traj.edit_height(height)

    if pitch is not None:
        new_traj = new_traj.edit_pitch(pitch)

    if scale is not None:
        new_traj = new_traj.scale(scale)

    print("=== New Trajectory Stats")
    print(f"height={new_traj.camtoworlds[:5, 2, 3]}")

    if pcd_path is not None:
        colmap = load_pcd_from_dir(pcd_path, False)
        print(f"  Loaded {len(colmap.points)} points")
        view_pcd_with_traj(colmap, [traj, new_traj])

    new_traj.save(output_path)


@click.command()
@click.argument("pcd-path", type=click.Path(path_type=Path, exists=True, dir_okay=True))
@click.option(
    "-t",
    "--traj-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option("--frustum-size", type=float, default=0.1, help="Size of camera frustums")
@click.option("--frustum-skip", type=int, default=20, help="Show every Nth frustum")
@click.option(
    "--sphere-radius",
    type=float,
    default=0.02,
    help="Radius of camera position spheres",
)
@click.option(
    "--sphere-skip", type=int, default=10, help="Show every Nth camera sphere"
)
def view(
    pcd_path: Path,
    traj_path: Path | None,
    frustum_size: float,
    frustum_skip: int,
    sphere_radius: float,
    sphere_skip: int,
):
    """
    Visualize a trajectory with a point cloud.

    Args:
        pcd_path: Path to point cloud file (.ply or .pcd)
        traj_path: Path to trajectory file (.npz)
        frustum_size: Size of camera frustum visualization
        frustum_skip: Show every Nth camera frustum
        sphere_radius: Radius of camera position spheres
        sphere_skip: Show every Nth camera position sphere
    """

    # Load point cloud
    print(f"Loading point cloud from {pcd_path}...")
    colmap = load_pcd_from_dir(pcd_path, False)
    print(f"  Loaded {len(colmap.points)} points")

    # Load trajectory
    trajs = None
    if traj_path is not None:
        trajs = [TrajectoryData.load(traj_path)]

    view_pcd_with_traj(
        colmap, trajs, frustum_skip=frustum_skip, sphere_skip=sphere_skip
    )


@click.command()
@click.argument("input-path", type=click.Path(path_type=Path, dir_okay=False, exists=True))
@click.option("-o", "--output-path", type=click.Path(path_type=Path, exists=False), required=True)
def fix_flip(input_path: Path, output_path: Path):
    """
    Fix left-right flip in trajectories generated with old generate_trajectory.py.

    This corrects trajectories that were generated before the fix where the right
    vector was computed incorrectly, causing rendered videos to be left-right flipped.

    Example:
        python trajectory_data.py fix-flip old_traj.npz -o fixed_traj.npz
    """
    print(f"Loading trajectory from {input_path}...")
    traj = TrajectoryData.load(input_path)

    print("\nApplying left-right flip correction...")
    fixed_traj = traj.fix_left_right_flip()

    print(f"\nSaving corrected trajectory to {output_path}...")
    fixed_traj.save(output_path)

    print("\n✓ Done! The corrected trajectory should now render without left-right flip.")
    print("  Test by rendering with: render_trajectory.py --traj-file " + str(output_path))


@click.group()
def main():
    pass


main.add_command(view, name="view")
main.add_command(edit_traj, name="edit")
main.add_command(fix_flip, name="fix-flip")

if __name__ == "__main__":
    main()
