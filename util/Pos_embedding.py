import torch
import torch.nn as nn
import numpy as np


class LocPositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=3660):
        super(LocPositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))  # 替换pos行，odds列的数据
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)  # (1,3660,d_model)

    def forward(self, S):
        pos = self.pos_enc_mat[:, :S, :]  # 位置矩阵与特征矩阵直接相加
        return pos  # (6,6,C*H*W)
