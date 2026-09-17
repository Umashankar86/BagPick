"""Pure geometric operations in original-frame pixels."""
from math import hypot

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def center(box: Box) -> Point:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def contains(box: Box, point: Point) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def overlap(box: Box, roi: Box) -> float:
    """Fraction of the backpack area covered by the placement ROI."""
    return intersection(box, roi) / max(area(box), 1e-9)


def iou(a: Box, b: Box) -> float:
    common = intersection(a, b)
    return common / max(area(a) + area(b) - common, 1e-9)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def point_box_distance(point: Point, box: Box) -> float:
    return hypot(max(box[0] - point[0], 0, point[0] - box[2]),
                 max(box[1] - point[1], 0, point[1] - box[3]))


def box_distance(a: Box, b: Box) -> float:
    return hypot(max(a[0] - b[2], b[0] - a[2], 0),
                 max(a[1] - b[3], b[1] - a[3], 0))
