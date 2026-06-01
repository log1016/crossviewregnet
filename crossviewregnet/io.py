from pathlib import Path

import laspy
import numpy as np


def read_las_xyz(path):
    """Read XYZ coordinates from a LAS or LAZ file."""
    las = laspy.read(str(path))
    return np.column_stack((las.x, las.y, las.z)).astype(np.float32)


def index_point_clouds(input_dir):
    """Return a basename -> file path index for LAS/LAZ files."""
    input_dir = Path(input_dir)
    paths = list(input_dir.rglob("*.las")) + list(input_dir.rglob("*.laz"))
    return {path.stem: path for path in paths}
