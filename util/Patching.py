from einops.layers.torch import Rearrange


def patching_method(Q, K, V, B, T, C, NUM_H, NUM_W, SIZE_H, SIZE_W, METHOD):
    to_patch = Rearrange('b t c (h p_h) (w p_w) -> b (t h w) (c p_h p_w)', p_h=SIZE_H, p_w=SIZE_W)
    if METHOD == 'STTN':
        Q = Q.view(B, -1, C, NUM_H, SIZE_H, NUM_W, SIZE_W)
        Q = Q.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            B, -1, C * SIZE_H * SIZE_W)
        K = K.view(B, -1, C, NUM_H, SIZE_H, NUM_W, SIZE_W)
        K = K.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            B, -1, C * SIZE_H * SIZE_W)
        V = V.view(B, -1, C, NUM_H, SIZE_H, NUM_W, SIZE_W)
        V = V.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            B, -1, C * SIZE_H * SIZE_W)
    elif METHOD == 'UNFOLD':
        Q = Q.unfold(2, SIZE_H, SIZE_H).unfold(3, SIZE_W, SIZE_W)
        Q = Q.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, SIZE_W, SIZE_H)
        Q = Q.reshape(B, -1, C * SIZE_W * SIZE_H)

        K = K.unfold(2, SIZE_H, SIZE_H).unfold(3, SIZE_W, SIZE_W)
        K = K.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, SIZE_W, SIZE_H)
        K = K.reshape(B, -1, C * SIZE_W * SIZE_H)

        V = V.unfold(2, SIZE_H, SIZE_H).unfold(3, SIZE_W, SIZE_W)
        V = V.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, SIZE_W, SIZE_H)
        V = V.reshape(B, -1, C * SIZE_W * SIZE_H)
    else:
        H = NUM_H*SIZE_H
        W = NUM_W*SIZE_W
        Q = Q.reshape(B, -1, C, H, W)
        K = K.reshape(B, -1, C, H, W)
        V = V.reshape(B, -1, C, H, W)
        Q = to_patch(Q)
        K = to_patch(K)
        V = to_patch(V)

    return Q, K, V
