def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    import numpy as np
    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    left = max(ax1, bx1)
    right = min(ax2, bx2)
    bottom = max(ay1, by1)
    top = min(ay2, by2)

    h = max(0, top - bottom)
    b = max(0, right - left)

    area_a = max(0, (ax2-ax1) * (ay2-ay1))
    area_b = max(0, (bx2-bx1) * (by2-by1))

    Intersection = h*b
    Union = area_a + area_b - Intersection

    if Union > 0.0 :
        return Intersection / Union
    else:
        return 0.0    