def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    import numpy as np 

    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b

    area_a = max(0, (x2a-x1a)*(y2a-y1a))
    area_b = max(0, (x2b-x1b)*(y2b-y1b))

    h = max(0, min(y2a,y2b)-max(y1a,y1b))
    b = max(0, min(x2a,x2b)-max(x1a,x1b))

    i = h * b
    u = area_a + area_b - i

    return float(i/u) if u>0.0 else 0.0 