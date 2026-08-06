from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIDUCIAL_MM_COORDS = {
    1: (0.57414, -96.9945),
    2: (-95.92, -0.112518),
    3: (95.92, 0.769928),
    4: (-0.246604, 96.9945),
}


def read_allowed_images(path: Path) -> set[str]:
    """Read the list of images allowed to be processed."""
    allowed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                allowed.add(value)
    return allowed


def parse_observations(path: Path) -> list[tuple[str, int, float, float, float, float]]:
    """Parse observation rows with the format: image_name,fiducial_id,x_px,y_px[,x_mm,y_mm]."""
    records: list[tuple[str, int, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = [value.strip() for value in row]
            if not values or values[0].startswith("#"):
                continue
            if len(values) not in (4, 6):
                continue

            try:
                image_name = values[0]
                fiducial_id = int(values[1])
                x_px = float(values[2])
                y_px = float(values[3])

                if len(values) == 6:
                    x_mm = float(values[4])
                    y_mm = float(values[5])
                elif fiducial_id in FIDUCIAL_MM_COORDS:
                    x_mm, y_mm = FIDUCIAL_MM_COORDS[fiducial_id]
                else:
                    continue

                records.append((image_name, fiducial_id, x_px, y_px, x_mm, y_mm))
            except (ValueError, KeyError):
                continue

    return records


def write_fiducials(output_path: Path, records: list[tuple[str, int, float, float, float, float]]) -> None:
    """Write the filtered fiducials to the requested format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# image_name,fiducial_id,x_px,y_px,x_mm,y_mm\n")
        for image_name, fiducial_id, x_px, y_px, x_mm, y_mm in records:
            if not 1 <= fiducial_id <= 4:
                continue
            handle.write(f"{image_name}," f"{fiducial_id},{x_px},{y_px},{x_mm},{y_mm}\n")


def _is_image_allowed(image_name: str, allowed_images: set[str]) -> bool:
    """Check if image is in allowed list, handling .tif extension variations."""
    if image_name in allowed_images:
        return True
    # Try adding .tif if not present
    if not image_name.endswith(".tif"):
        if f"{image_name}.tif" in allowed_images:
            return True
    # Try removing .tif if present
    if image_name.endswith(".tif"):
        if image_name[:-4] in allowed_images:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fiducials from observation rows")
    parser.add_argument("allowed_images", help="Path to a text file listing allowed image names")
    parser.add_argument("observations", help="Path to the observations CSV-like file")
    parser.add_argument("output", help="Path to the output fiducials file")
    args = parser.parse_args()

    allowed_images = read_allowed_images(Path(args.allowed_images))
    records = parse_observations(Path(args.observations))

    filtered_records = [
        (image_name, fiducial_id, x_px, y_px, x_mm, y_mm)
        for image_name, fiducial_id, x_px, y_px, x_mm, y_mm in records
        if _is_image_allowed(image_name, allowed_images)
    ]
    write_fiducials(Path(args.output), filtered_records)


if __name__ == "__main__":
    main()
