def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    import numpy as np 

    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios)

    stride = image_size // feature_size 

    res = []
  
    widths = np.outer(scales, np.sqrt(aspect_ratios)).flatten()
    heights = np.outer(scales, 1 / np.sqrt(aspect_ratios)).flatten()

    for i in range(feature_size):
      for j in range(feature_size):
        
        cx = (j+0.5) * stride
        cy = (i+0.5) * stride

        for w, h in zip(widths, heights):
          x1 = cx - w/2
          x2 = cy - h/2
          y1 = cx + w/2
          y2 = cy + h/2
          res.append([x1, x2, y1, y2])

    return res
        