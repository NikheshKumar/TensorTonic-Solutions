import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    import numpy as np 

    feature_map = np.asarray(feature_map)

    rois = np.asarray(rois)

    ans = []

    for r in rois:
        x1,y1,x2,y2 = r

        roi_h = y2-y1
        roi_w = x2-x1

        output = np.zeros((output_size, output_size), dtype=feature_map.dtype)

        for i in range(output_size):
            for j in range(output_size):
                h_start = int(math.floor(y1 + (i* roi_h // output_size)))
                h_end = int(math.ceil(y1 + ((i+1)* roi_h // output_size)))

                w_start = int(math.floor(x1 + (j* roi_w // output_size)))
                w_end = int(math.ceil(x1 + ((j+1)* roi_w // output_size)))

                if h_start == h_end and h_start < feature_map.shape[0]:
                    h_end += 1
                if w_start == w_end and w_start < feature_map.shape[1]:
                    w_end += 1

                if feature_map[h_start:h_end, w_start:w_end].size > 0:
                    output[i, j] = np.max(feature_map[h_start:h_end, w_start:w_end])
                else:
                    output[i, j] = 0

        ans.append(output.tolist())

    return ans
