import os
import pycolmap
import numpy as np
from typing import Dict, List, Tuple


def main(
    reconstruction_folder: str,
    bundler_out_file: str,
    out_list_file: str,
    output_folder: str
) -> Dict:
    """
    Parse COLMAP reconstruction folder and related files.
    
    Args:
        reconstruction_folder: Path to COLMAP reconstruction folder
        bundler_out_file: Path to bundler out file
        out_list_file: Path to out.list.txt file
        output_folder: Output folder for results
    
    Returns:
        Dictionary containing parsed reconstruction data
    """

    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    out_3Dpoints_path = os.path.join(output_folder, "triangulated_points.txt")
    out_image_pos_path = os.path.join(output_folder, "image_poses.txt")
    
    # Parse image list
    image_list = parse_image_list(out_list_file)
    image_dict = {idx: name for idx, name in enumerate(image_list)}
    
    # Parse bundler output
    bundler_data = parse_bundler_out(bundler_out_file)
    
    # Parse COLMAP reconstruction (cameras, images, points3D)
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(reconstruction_folder)
    print(reconstruction.summary())

    cameras = reconstruction.cameras
    images = reconstruction.images
    
    #print(cameras[1])
    #print(images[1])
    #print(images[1].cam_from_world());quit()

    triangulation_options = pycolmap.EstimateTriangulationOptions()

    with open(out_image_pos_path, 'w') as out_image_pos_file:
        for i in images:
            center = images[i].projection_center()
            out_image_pos_file.write(f"{center[0]} {center[1]} {center[2]}\n")

    with open(out_3Dpoints_path, 'w') as out_3Dpoints_file:
        for point3D in bundler_data["points"]:
            print(point3D)
            points_for_triang = np.empty((0, 2))
            cameras_for_triang = []
            cams_from_world_for_triang = []

            for view in point3D['views']:
                image_id = view['camera_idx'] + 1
                colmap_image = images[image_id]
                cams_from_world_for_triang.append(colmap_image.cam_from_world())
                width = colmap_image.camera.width
                height = colmap_image.camera.height
                x = view['x'] + width/2
                y = height/2 - view['y']
                points_for_triang = np.vstack((points_for_triang, np.array([[x, y]])))
                cameras_for_triang.append(colmap_image.camera)


            point3D = pycolmap.estimate_triangulation(
                points=points_for_triang,
                cams_from_world=cams_from_world_for_triang,
                cameras=cameras_for_triang,
                options=triangulation_options,
            )
            print(point3D)
            if point3D is not None:
                out_3Dpoints_file.write(f"{point3D['xyz'][0]} {point3D['xyz'][1]} {point3D['xyz'][2]}\n")

            #quit()
    
    return reconstruction


def parse_image_list(filepath: str) -> List[str]:
    """Parse out.list.txt file containing image list."""
    images = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                images.append(line)
    return images


def parse_bundler_out(filepath: str) -> Dict:
    """Parse Bundler ``bundle.out`` file (v0.3 format).

    Expected layout:
    - Header: ``# Bundle file vX.Y``
    - Counts: ``<num_cameras> <num_points>``
    - Camera blocks (5 lines each):
      ``f k1 k2``, 3x3 rotation matrix (3 lines), ``tx ty tz``
    - Point blocks (3 lines each):
      ``X Y Z``, ``R G B``, visibility list
    """

    with open(filepath, 'r') as f:
        # Ignore blank lines to tolerate minor formatting differences.
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Invalid Bundler file: {filepath}")

    version = lines[0]
    if not version.startswith('# Bundle file'):
        raise ValueError(
            f"Unexpected Bundler header '{version}' in {filepath}"
        )

    try:
        num_cameras, num_points = map(int, lines[1].split())
    except Exception as exc:
        raise ValueError(
            f"Could not parse camera/point counts from line: '{lines[1]}'"
        ) from exc

    idx = 2
    cameras: Dict[int, Dict] = {}
    for cam_id in range(num_cameras):
        if idx + 4 >= len(lines):
            raise ValueError("Unexpected end of file while parsing camera blocks")

        try:
            f, k1, k2 = map(float, lines[idx].split())
            r1 = list(map(float, lines[idx + 1].split()))
            r2 = list(map(float, lines[idx + 2].split()))
            r3 = list(map(float, lines[idx + 3].split()))
            t = list(map(float, lines[idx + 4].split()))
        except Exception as exc:
            raise ValueError(
                f"Malformed camera block {cam_id} near line {idx + 1}"
            ) from exc

        if len(r1) != 3 or len(r2) != 3 or len(r3) != 3 or len(t) != 3:
            raise ValueError(f"Invalid camera block dimensions for camera {cam_id}")

        cameras[cam_id] = {
            'focal_length': f,
            'k1': k1,
            'k2': k2,
            'R': [r1, r2, r3],
            't': t,
        }
        idx += 5

    points: List[Dict] = []
    for point_id in range(num_points):
        if idx + 2 >= len(lines):
            raise ValueError("Unexpected end of file while parsing point blocks")

        try:
            xyz = list(map(float, lines[idx].split()))
            rgb = list(map(int, lines[idx + 1].split()))
            view_tokens = lines[idx + 2].split()
        except Exception as exc:
            raise ValueError(
                f"Malformed point block {point_id} near line {idx + 1}"
            ) from exc

        if len(xyz) != 3 or len(rgb) != 3:
            raise ValueError(f"Invalid XYZ/RGB dimensions for point {point_id}")

        if not view_tokens:
            raise ValueError(f"Missing visibility list for point {point_id}")

        num_views = int(view_tokens[0])
        expected_tokens = 1 + 4 * num_views
        if len(view_tokens) != expected_tokens:
            raise ValueError(
                "Invalid visibility list length for point "
                f"{point_id}: expected {expected_tokens}, got {len(view_tokens)}"
            )

        views = []
        token_idx = 1
        for _ in range(num_views):
            camera_idx = int(view_tokens[token_idx])
            keypoint_idx = int(view_tokens[token_idx + 1])
            x = float(view_tokens[token_idx + 2])
            y = float(view_tokens[token_idx + 3])
            views.append({
                'camera_idx': camera_idx,
                'keypoint_idx': keypoint_idx,
                'x': x,
                'y': y,
            })
            token_idx += 4

        points.append({
            'id': point_id,
            'xyz': xyz,
            'rgb': rgb,
            'num_views': num_views,
            'views': views,
        })
        idx += 3

    return {
        'version': version,
        'num_cameras': num_cameras,
        'num_points': num_points,
        'cameras': cameras,
        'points': points,
    }





if __name__ == "__main__":
    reconstruction = main(
        reconstruction_folder=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\reconstruction",
        bundler_out_file=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\bundler.out",
        out_list_file=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\bundler.out.list.txt",
        output_folder=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\out"
    )