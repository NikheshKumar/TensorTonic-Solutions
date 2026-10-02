import math

def sobel_edges(image: list) -> list:
    """
    Returns the zero-padded Sobel gradient magnitude at every pixel.
    """
    # Write code here

    H = len(image)
    W = len(image[0])
    
    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]] 
    
    Ky = [[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]] 

    Gx = [[0.0] * W for _ in range(H)]
    Gy = [[0.0] * W for _ in range(H)]

    P = [[0.0] * (W + 2) for _ in range(H + 2)]
    for i in range(H):
        for j in range(W):
            P[i + 1][j + 1] = image[i][j]

    for i in range(H):
        row = []
        for j in range(W):
            for a in range(3):
                for b in range(3):
                    Gx[i][j] += Kx[a][b] * P[i+a][j+b]
                    Gy[i][j] += Ky[a][b] * P[i+a][j+b]


    G = [[math.sqrt(Gx[i][j] ** 2 + Gy[i][j] ** 2) for j in range(W)] for i in range(H)]


    return G