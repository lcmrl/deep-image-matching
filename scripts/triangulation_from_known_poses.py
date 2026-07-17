import os
import pycolmap
import pyceres
import pycolmap.cost_functions

import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


def run_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    output_folder: str
) -> None:
    """Run bundle adjustment on the given reconstruction and 3D points."""


    refined_poses_path = os.path.join(output_folder, "refined_poses.txt")
    refined_poses_file = open(refined_poses_path, 'w')

    #prob = pyceres.Problem()
    #loss = pyceres.TrivialLoss()
#
    #for im in reconstruction.images.values():
    #    cam = reconstruction.cameras[im.camera_id]
    #    print(im)
    #    print(cam)
    #    quit()

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_config = pycolmap.BundleAdjustmentConfig()

    ba_options.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba_options.loss_function_scale = 1.0
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    summary = bundle_adjuster.solve()
    print(summary)

    reconstruction.write(output_folder)


    for img in reconstruction.images.values():
        print(img);quit()
        pose = img.projection_center()
        refined_poses_file.write(f"{pose[0]} {pose[1]} {pose[2]}\n")

    refined_poses_file.close()



def triangulate_from_known_poses(
    reconstruction_folder: str,
    bundler_out_file: str,
    out_list_file: str,
    output_folder: str
) -> pycolmap.Reconstruction:
    """
    Parse COLMAP reconstruction folder and related files.
    
    Args:
        reconstruction_folder: Path to COLMAP reconstruction folder
        bundler_out_file: Path to bundler out file
        out_list_file: Path to out.list.txt file
        output_folder: Output folder for results
    
    Returns:
        Tuple containing the parsed reconstruction and 3D points
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
    
    new_reconstruction = pycolmap.Reconstruction()

    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = {}

    triangulation_options = pycolmap.EstimateTriangulationOptions()

    for camera in cameras.values():
        new_reconstruction.add_camera(deepcopy(camera))

    colmap_rig = pycolmap.Rig(
        {
            "rig_id": 1,
        }
    )
    sensor1 = pycolmap.sensor_t(
        {
            "type": pycolmap.SensorType.CAMERA, 
            "id": 1,  # Use camera_id 1 (first camera)
         }
         )
    colmap_rig.add_ref_sensor(sensor1)
    new_reconstruction.add_rig(colmap_rig)

    

    with open(out_image_pos_path, 'w') as out_image_pos_file:
        for i in range(1,len(images)+1):
            data_t = pycolmap.data_t({"sensor_id": sensor1, "id": i})
            center = images[i].projection_center()
            out_image_pos_file.write(f"{center[0]} {center[1]} {center[2]}\n")
            init_cam_from_world = images[i].cam_from_world()

            # Add frame
            new_frame = pycolmap.Frame()
            new_frame.frame_id = i
            new_frame.rig_id = 1
            new_frame.add_data_id(data_t)
            new_reconstruction.add_frame(new_frame)
            print("frame data id", new_frame.num_data_ids)
            
            # Add image
            new_image = pycolmap.Image({
                "image_id": images[i].image_id,
                "camera_id": images[i].camera_id,
                "frame_id": images[i].frame_id,
            })
            print("image data id", new_image.data_id)
            #new_reconstruction.add_image(new_image)
            new_reconstruction.add_image_with_trivial_frame(new_image, init_cam_from_world)
    

    with open(out_3Dpoints_path, 'w') as out_3Dpoints_file:
        for n,point3D in enumerate(bundler_data["points"]):
            print(point3D)
            points_for_triang = np.empty((0, 2))
            cameras_for_triang = []
            cams_from_world_for_triang = []
            track = pycolmap.Track()

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
                

                img = new_reconstruction.image(image_id)
                img.points2D.append(pycolmap.Point2D(np.array([x, y])))
                observ_id = len(img.points2D) - 1
                #img.set_point3D_for_point2D(observ_id, n)
                track.add_element(image_id, observ_id)
            


            point3Dxyz = pycolmap.estimate_triangulation(
                points=points_for_triang,
                cams_from_world=cams_from_world_for_triang,
                cameras=cameras_for_triang,
                options=triangulation_options,
            )
            #print(point3Dxyz)
            if point3Dxyz is not None:
                out_3Dpoints_file.write(f"{point3Dxyz['xyz'][0]} {point3Dxyz['xyz'][1]} {point3Dxyz['xyz'][2]}\n")
                points3D[point3D['id']] = point3Dxyz
                try:
                    new_reconstruction.add_point3D(
                        point3Dxyz['xyz'].reshape(3,1),
                        track,
                    )
                except Exception as e:
                    print(f"Error adding point3D: {e}")
    
    return new_reconstruction


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
    output_folder = r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\out"
    new_reconstruction = triangulate_from_known_poses(
        reconstruction_folder=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\reconstruction",
        bundler_out_file=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\bundler.out",
        out_list_file=r"C:\Users\threedom\Desktop\lcmrl-github\deep-image-matching\assets\pytest\results_superpoint+lightglue_bruteforce_quality_high\bundler.out.list.txt",
        output_folder=output_folder
    )

    run_bundle_adjustment(new_reconstruction, output_folder)