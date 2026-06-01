import numpy as np
from sklearn.cluster import DBSCAN


def points_to_images(
    points,
    resolution=256,
    num_side_views=4,
    max_points=500000,
    dbh_min=1.0,
    dbh_max=1.5,
    dbh_dbscan_eps=0.10,
    dbh_dbscan_min_samples=10,
    max_dbh_points=30000,
):
    """Convert one tree point cloud into multi-view projection images.

    View order is: top, side views, bottom, DBH slice.
    """
    points = np.asarray(points, dtype=np.float32)
    views = np.zeros((num_side_views + 3, resolution, resolution), dtype=np.float32)
    if points.size == 0:
        return views

    views[num_side_views + 2] = dbh_section_view(
        points,
        resolution=resolution,
        dbh_min=dbh_min,
        dbh_max=dbh_max,
        dbscan_eps=dbh_dbscan_eps,
        dbscan_min_samples=dbh_dbscan_min_samples,
        max_dbh_points=max_dbh_points,
    )

    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    points = points - np.median(points, axis=0)
    max_abs = float(np.max(np.abs(points)))
    if max_abs > 1e-6:
        points = points / max_abs

    views[0] = top_view(points, resolution=resolution, inverse=False)
    for view_idx, degree in enumerate(np.linspace(0, 360, num_side_views, endpoint=False)):
        rad = np.deg2rad(degree)
        rot = np.array(
            [
                [np.cos(rad), -np.sin(rad), 0.0],
                [np.sin(rad), np.cos(rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotated = points @ rot.T
        views[view_idx + 1] = side_view(rotated, resolution=resolution)

    views[num_side_views + 1] = top_view(points, resolution=resolution, inverse=True)
    return views


def top_view(points, resolution=256, inverse=False):
    if points.shape[0] == 0:
        return np.zeros((resolution, resolution), dtype=np.float32)
    return _rasterize(
        axis_a=points[:, 0],
        axis_b=points[:, 1],
        depth=points[:, 2],
        resolution=resolution,
        reduce="min" if inverse else "max",
    )


def side_view(points, resolution=256):
    if points.shape[0] == 0:
        return np.zeros((resolution, resolution), dtype=np.float32)
    return _rasterize(
        axis_a=points[:, 0],
        axis_b=points[:, 2],
        depth=points[:, 1],
        resolution=resolution,
        reduce="max",
    )


def dbh_section_view(
    points,
    resolution=256,
    dbh_min=1.0,
    dbh_max=1.5,
    dbscan_eps=0.10,
    dbscan_min_samples=10,
    max_dbh_points=30000,
):
    section = points[(points[:, 2] > dbh_min) & (points[:, 2] < dbh_max)]
    if section.shape[0] <= 50:
        return np.zeros((resolution, resolution), dtype=np.float32)

    if section.shape[0] > max_dbh_points:
        idx = np.random.choice(section.shape[0], max_dbh_points, replace=False)
        section = section[idx]

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(section[:, :2])
    valid_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) > 0:
        section = section[labels == valid_labels[np.argmax(counts)]]

    if section.shape[0] <= 1:
        return np.zeros((resolution, resolution), dtype=np.float32)

    section = section - np.median(section, axis=0)
    max_abs = float(np.max(np.abs(section)))
    if max_abs > 1e-6:
        section = section / max_abs
    return top_view(section, resolution=resolution, inverse=False)


def _rasterize(axis_a, axis_b, depth, resolution, reduce):
    axis_a = np.asarray(axis_a)
    axis_b = np.asarray(axis_b)
    depth = np.asarray(depth, dtype=np.float32)

    range_a = float(axis_a.max() - axis_a.min())
    range_b = float(axis_b.max() - axis_b.min())
    cell_size = max(range_a, range_b) / float(resolution)
    if cell_size < 1e-8:
        return np.zeros((resolution, resolution), dtype=np.float32)

    pixel_a = ((axis_a - axis_a.min()) / cell_size).astype(np.int64)
    pixel_b = ((axis_b - axis_b.min()) / cell_size).astype(np.int64)
    pixel_a = np.clip(pixel_a, 0, resolution - 1)
    pixel_b = np.clip(pixel_b, 0, resolution - 1)

    fill_value = float(depth.min())
    if reduce == "max":
        image = np.full((resolution, resolution), -np.inf, dtype=np.float32)
        np.maximum.at(image, (pixel_a, pixel_b), depth)
        image[~np.isfinite(image)] = fill_value
    elif reduce == "min":
        image = np.full((resolution, resolution), np.inf, dtype=np.float32)
        np.minimum.at(image, (pixel_a, pixel_b), depth)
        image[~np.isfinite(image)] = fill_value
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")
    return image
