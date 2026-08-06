from __future__ import annotations

import csv
import gc
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

input_folder = r"G:\Projects\Luca\\Experiments\all_images\images"
output_folder = r"G:\Projects\Luca\\Experiments\all_images\normalized_images"
fiducials_file = r"G:\Projects\Luca\\Experiments\all_images\fiducials.txt"
scanner_resolution = 25.4 / 2400  # mm/pixel
method = "original_fiducials"


def parse_fiducials(fiducials_path: str | Path) -> list[dict[str, float | str | int]]:
    """Parse a CSV-like fiducials file with the format:
    image_name,fiducial_id,x_px,y_px,x_mm,y_mm
    """
    fiducials_path = Path(fiducials_path)
    records: list[dict[str, float | str | int]] = []

    with fiducials_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = [value.strip() for value in row]
            if not values or values[0].startswith("#"):
                continue
            if len(values) != 6:
                raise ValueError(f"Expected 6 values per fiducial entry, got {len(values)}: {row}")

            image_name, fiducial_id, x_px, y_px, x_mm, y_mm = values
            records.append(
                {
                    "image_name": image_name,
                    "fiducial_id": int(fiducial_id),
                    "x_px": float(x_px),
                    "y_px": float(y_px),
                    "x_mm": float(x_mm),
                    "y_mm": float(y_mm),
                }
            )

    return records


def _read_image_as_rgb(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = src.read()

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    else:
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def _order_corners_for_output(corners: np.ndarray) -> np.ndarray:
    """Return corners in the order expected by the homography: top-left, top-right, bottom-right, bottom-left."""
    if corners.shape != (4, 2):
        raise ValueError("Expected 4 corners")

    center = np.mean(corners, axis=0)
    ordered = []
    for corner in corners:
        if corner[0] <= center[0] and corner[1] <= center[1]:
            ordered.append((0, corner))
        elif corner[0] > center[0] and corner[1] <= center[1]:
            ordered.append((1, corner))
        elif corner[0] > center[0] and corner[1] > center[1]:
            ordered.append((2, corner))
        else:
            ordered.append((3, corner))

    ordered.sort(key=lambda item: item[0])
    return np.array([item[1] for item in ordered], dtype=float)


def compute_square_corners(points: dict[int, np.ndarray]) -> np.ndarray:
    """Return the 4 square corners from four edge-midpoint points labeled 1..4."""
    if set(points) != {1, 2, 3, 4}:
        raise ValueError("Expected midpoint points with ids 1, 2, 3, and 4")

    center = np.mean([points[1], points[2], points[3], points[4]], axis=0)
    axis_13 = (points[3] - points[1]) / 2.0
    axis_24 = (points[4] - points[2]) / 2.0

    corners = np.array(
        [
            center + axis_13 + axis_24,
            center + axis_13 - axis_24,
            center - axis_13 - axis_24,
            center - axis_13 + axis_24,
        ],
        dtype=float,
    )
    return _order_corners_for_output(corners)


def compute_output_size(dist_p1_p3_mm: float, dist_p2_p4_mm: float, scale: float = 1.0) -> tuple[int, int]:
    """Compute output width/height in pixels from the measured fiducial distances."""
    width_px = max(1, int(round((dist_p2_p4_mm / scanner_resolution) * scale)))
    height_px = max(1, int(round((dist_p1_p3_mm / scanner_resolution) * scale)))
    return width_px, height_px


def compute_homography_points(
    points: dict[int, np.ndarray],
    output_shape: tuple[int, int],
    method_name: str = method,
) -> tuple[np.ndarray, np.ndarray]:
    """Build source/destination points for the selected homography method."""
    width_px, height_px = output_shape

    if method_name == "original_fiducials":
        src_points = np.array([points[1], points[2], points[3], points[4]], dtype=float)
        dst_points = np.array(
            [
                [width_px / 2.0, 0.0],
                [width_px - 1, height_px / 2.0],
                [width_px / 2.0, height_px - 1],
                [0.0, height_px / 2.0],
            ],
            dtype=float,
        )
        return src_points, dst_points

    if method_name != "extended_fiducials":
        raise ValueError(f"Unsupported method: {method_name}")

    src_points = compute_square_corners(points)
    dst_points = np.array(
        [
            [0.0, 0.0],
            [width_px - 1, 0.0],
            [width_px - 1, height_px - 1],
            [0.0, height_px - 1],
        ],
        dtype=float,
    )
    return src_points, dst_points


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute a 3x3 homography mapping src_points to dst_points."""
    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("Expected 4 source and 4 destination points")

    homography, _ = cv2.findHomography(
        src_points.astype(np.float32),
        dst_points.astype(np.float32),
        method=cv2.RANSAC,
    )
    return homography


def warp_image_with_homography(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp an image to a new canvas using a homography."""
    h = compute_homography(src_points, dst_points)
    height, width = output_shape
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    warped_bgr = cv2.warpPerspective(image_bgr, h, (width, height), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)


def write_tiff_image(image: np.ndarray, output_path: Path) -> None:
    """Write an image array to disk as a GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=image.shape[2],
        dtype=image.dtype,
    ) as dst:
        dst.write(image.transpose(2, 0, 1))


def visualize_fiducials(
    input_folder: str | Path,
    fiducials_file: str | Path,
    output_folder: str | Path,
) -> list[Path]:
    """Create one visualization image per input image with fiducial points and square corners overlaid."""
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Starting visualization from {input_dir}")
    print(f"[DEBUG] Loading fiducials from {fiducials_file}")
    records = parse_fiducials(fiducials_file)
    print(f"[DEBUG] Loaded {len(records)} fiducial records")
    grouped_records: dict[str, list[dict[str, float | str | int]]] = {}
    for record in records:
        grouped_records.setdefault(str(record["image_name"]), []).append(record)
    print(f"[DEBUG] Found {len(grouped_records)} unique images with fiducials")

    outputs: list[Path] = []
    for image_path in sorted(input_dir.iterdir()):
        if not image_path.is_file():
            continue

        try:
            image = _read_image_as_rgb(image_path)

            image_name = image_path.name
            image_records = grouped_records.get(image_name, [])
            if not image_records:
                image_records = grouped_records.get(image_path.stem, [])
            if not image_records:
                continue

            midpoints_px = {
                int(record["fiducial_id"]): np.array([float(record["x_px"]), float(record["y_px"])])
                for record in image_records
                if int(record["fiducial_id"]) in {1, 2, 3, 4}
            }

            midpoints_mm = {
                int(record["fiducial_id"]): np.array([float(record["x_mm"]), float(record["y_mm"])])
                for record in image_records
                if int(record["fiducial_id"]) in {1, 2, 3, 4}
            }

            if set(midpoints_px) != {1, 2, 3, 4}:
                continue

            dist_p1_p3_px = float(np.linalg.norm(midpoints_px[3] - midpoints_px[1]))
            dist_p2_p4_px = float(np.linalg.norm(midpoints_px[4] - midpoints_px[2]))
            dist_p1_p3_mm = float(np.linalg.norm(midpoints_mm[3] - midpoints_mm[1]))
            dist_p2_p4_mm = float(np.linalg.norm(midpoints_mm[4] - midpoints_mm[2]))
            print(
                f"{image_name}: dist_p1_p3 = {dist_p1_p3_px:.3f}px | {dist_p1_p3_mm:.3f}mm; "
                f"dist_p2_p4 = {dist_p2_p4_px:.3f}px | {dist_p2_p4_mm:.3f}mm;"
                f"ratio_px = {dist_p1_p3_px / dist_p2_p4_px:.3f}; ratio_mm = {dist_p1_p3_mm / dist_p2_p4_mm:.3f}"
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_title(image_name)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.axis("off")

            record_points = {
                int(record["fiducial_id"]): np.array([float(record["x_px"]), float(record["y_px"])])
                for record in image_records
                if int(record["fiducial_id"]) in {1, 2, 3, 4}
            }
            for fiducial_id, point in record_points.items():
                ax.scatter(point[0], point[1], color="red", s=5, marker="o")
                ax.text(point[0] + 2, point[1] + 2, str(fiducial_id), color="red", fontsize=7, weight="bold")

            corners = compute_square_corners(midpoints_px)
            square_edges = np.array(
                [
                    [corners[0], corners[1]],
                    [corners[1], corners[2]],
                    [corners[2], corners[3]],
                    [corners[3], corners[0]],
                ],
                dtype=float,
            )
            for edge in square_edges:
                ax.plot(edge[:, 0], edge[:, 1], color="lime", linewidth=0.5)
            ax.plot([record_points[1][0], record_points[3][0]], [record_points[1][1], record_points[3][1]], color="red", linewidth=0.5)
            ax.plot([record_points[2][0], record_points[4][0]], [record_points[2][1], record_points[4][1]], color="red", linewidth=0.5)
            ax.scatter(corners[:, 0], corners[:, 1], color="lime", s=5, marker="x")

            width_px, height_px = compute_output_size(dist_p1_p3_mm, dist_p2_p4_mm)
            src_points, dst_points = compute_homography_points(midpoints_px, (width_px, height_px), method)
            rectified = warp_image_with_homography(
                image.astype(np.uint8),
                src_points,
                dst_points,
                (height_px, width_px),
            )
            rectified_path = output_dir / f"{image_path.stem}_rectified.tif"
            write_tiff_image(rectified, rectified_path)

            output_path = output_dir / f"{image_path.stem}_fiducials.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            del image, rectified, fig
            gc.collect()
            outputs.append(output_path)
            outputs.append(rectified_path)

        except Exception as e:
            print(f"Error processing {image_path.name}: {type(e).__name__}: {e}")
            continue

    return outputs


if __name__ == "__main__":
    visualize_fiducials(input_folder, fiducials_file, output_folder)