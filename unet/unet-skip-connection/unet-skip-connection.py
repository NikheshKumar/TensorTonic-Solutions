import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    
    encoder_features = np.asarray(encoder_features, dtype=np.float64)
    decoder_features = np.asarray(decoder_features, dtype=np.float64)

    B, H_e, W_e, C_e = encoder_features.shape
    B, H_d, W_d, C_d = decoder_features.shape

    diff_H, diff_W = H_e - H_d, W_e - W_d
    start_H, start_W = diff_H//2, diff_W//2 

    cropped_encoder = encoder_features[:,start_H:start_H+H_d,start_W:start_W+W_d,:] 
    output = np.concatenate([cropped_encoder, decoder_features], axis=3)

    return output
