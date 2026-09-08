def make_vgg_config(variant: str) -> list:
    """
    Returns the canonical VGG layer configuration as a new list.
    """
    d = {"vgg11":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
        "vgg13":[64, 64, "M",128, 128, "M",256,256,"M",512,512,"M",512,512,"M"],
        "vgg16":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
        "vgg19":[64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"]}

    variant = variant.lower()

    return d[variant]