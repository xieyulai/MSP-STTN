import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummaryX import summary
import numpy as np
import argparse
from util.Patching import patching_method
from util.Pos_embedding import LocPositionalEncoder


class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, residual=True, bn=True,
                 activation='LeakyReLU'):

        super().__init__()
        self.is_bn = bn
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()

    def forward(self, x):

        res = self.residual(x)

        x = self.conv(x)

        if self.is_bn:
            x = self.bn(x)

        x = x + res
        x = self.activation(x)

        return x


class Attention(nn.Module):

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m:
            scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class Patch_Transformer(nn.Module):

    def __init__(self, length, encoding_w, encoding_h, encoding_dim, patch_size_w, patch_size_h, sub_embedding_dim,
                 is_mask=0, PATCH_METHOD='UNFOLD', Debugging=0):

        super().__init__()

        self.is_mask = is_mask
        self.length = length
        self.patch_method = PATCH_METHOD
        self.Debugging = Debugging

        self.encoding_w = encoding_w  # 32
        self.encoding_h = encoding_h  # 32

        self.patch_size_w = patch_size_w  # 2
        self.patch_size_h = patch_size_h  # 16

        self.patch_num_w = self.encoding_w // self.patch_size_w  # 16
        self.patch_num_h = self.encoding_h // self.patch_size_h  # 2

        # 1D vector
        mid_dim = sub_embedding_dim * self.patch_size_w * self.patch_size_h  # 256*2*16

        self.embedding_Q = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)
        self.embedding_K = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)
        self.embedding_V = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)

        if is_mask:
            self.multihead_attn = Attention()
        else:
            self.multihead_attn = nn.MultiheadAttention(mid_dim, num_heads=1)

    def forward(self, c, q, mask):

        # [12, 768, 32, 32]
        B_T, C, H, W = c.shape
        T = self.length
        B = B_T // T

        encoding_w = self.encoding_w  # 32
        encoding_h = self.encoding_h  # 32

        Q = self.embedding_Q(q)
        K = self.embedding_K(c)  # (24, 32, 32, 32)
        V = self.embedding_V(c)

        # B,C//num
        C = Q.shape[1]
        Q, K, V = patching_method(Q, K, V, B, T, C, self.patch_num_h, self.patch_num_w, self.patch_size_h,
                                  self.patch_size_w, self.patch_method)

        if self.is_mask:
            attn_output, atten_output_weight = self.multihead_attn(Q, K, V, None)
            x = attn_output
        else:
            Q = Q.permute(1, 0, 2)
            K = K.permute(1, 0, 2)
            V = V.permute(1, 0, 2)
            attn_output, atten_output_weight = self.multihead_attn(Q, K, V)
            x = attn_output.permute(1, 0, 2)

        x = x.reshape(B_T, -1, encoding_h, encoding_w)
        if self.Debugging: print('- patch 2D \t\t', x.shape)

        return x


class Encoder(nn.Module):

    def __init__(self, input_channels, encoding_dim):
        super().__init__()

        self.conv1 = Conv_block(input_channels, encoding_dim // 4, kernel_size=3, stride=1, dilation=1,
                                residual=False)
        self.conv2 = Conv_block(encoding_dim // 4, encoding_dim // 4, kernel_size=3, stride=1, dilation=1,
                                residual=True)
        self.conv3 = Conv_block(encoding_dim // 4, encoding_dim // 2, kernel_size=3, stride=1, dilation=1,
                                residual=True)
        self.conv4 = Conv_block(encoding_dim // 2, encoding_dim, kernel_size=3, stride=1, dilation=1,
                                residual=True)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c4, c3, c2, c1


class Decoder(nn.Module):

    def __init__(self, output_channels, encoding_dim, Using_skip, Activation):

        super().__init__()

        self.Using_skip = Using_skip

        self.conv5 = Conv_block(encoding_dim, encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
        self.conv6 = Conv_block(encoding_dim // 2 if Using_skip else encoding_dim // 4,
                                encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
        self.conv7 = Conv_block(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
        self.conv8 = Conv_block(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                output_channels, kernel_size=3, stride=1, dilation=1,
                                activation=Activation)

    def forward(self, inp):

        # [12,768,32,32] [12,64,32,32]
        c4, c3, c2, c1 = inp

        c5 = self.conv5(c4)
        if self.Using_skip:
            c5 = torch.cat([c5, c3], dim=1)

        c6 = self.conv6(c5)

        if self.Using_skip:
            c6 = torch.cat([c6, c2], dim=1)

        c7 = self.conv7(c6)
        if self.Using_skip:
            c7 = torch.cat([c7, c1], dim=1)

        c8 = self.conv8(c7)

        return c8


class DecoderRe(nn.Module):

    def __init__(self, output_channels, encoding_dim, Using_skip, Is_trans, Activation):

        super().__init__()

        self.Using_skip = Using_skip
        self.is_trans = Is_trans

        self.conv5 = Conv_block(encoding_dim, encoding_dim // 2, kernel_size=3, stride=1, dilation=1)
        self.tran5 = Conv_block(encoding_dim, encoding_dim // 2, kernel_size=1, stride=1, dilation=1, residual=False)

        self.conv6 = Conv_block(encoding_dim // 2 if Is_trans else encoding_dim, encoding_dim // 4, kernel_size=3,
                                stride=1, dilation=1)
        self.tran6 = Conv_block(encoding_dim // 2, encoding_dim // 4, kernel_size=1, stride=1, dilation=1,
                                residual=False)

        self.conv7 = Conv_block(encoding_dim // 4 if Is_trans else encoding_dim // 2, encoding_dim // 4, kernel_size=3,
                                stride=1, dilation=1)
        self.tran7 = Conv_block(encoding_dim // 2, encoding_dim // 4, kernel_size=1, stride=1, dilation=1,
                                residual=False)

        self.conv8 = Conv_block(encoding_dim // 4 if Is_trans else encoding_dim // 2, output_channels, kernel_size=3,
                                stride=1, dilation=1,
                                activation=Activation)

    def forward(self, inp):

        # [12,768,32,32] [12,64,32,32]
        c4, c3, c2, c1 = inp

        c5 = self.conv5(c4)
        if self.Using_skip:
            c5 = torch.cat([c5, c3], dim=1)
            if self.is_trans:
                c5 = self.tran5(c5)
            else:
                c5 = c5

        c6 = self.conv6(c5)
        if self.Using_skip:
            c6 = torch.cat([c6, c2], dim=1)
            if self.is_trans:
                c6 = self.tran6(c6)
            else:
                c6 = c6

        c7 = self.conv7(c6)
        if self.Using_skip:
            c7 = torch.cat([c7, c1], dim=1)
            if self.is_trans:
                c7 = self.tran7(c7)
            else:
                c7 = c5

        c8 = self.conv8(c7)

        return c8


class Multi_patch_transfomer(nn.Module):

    def __init__(self, Patch_list, length, cnn_encoding_w, cnn_encoding_h, cnn_encoding_dim, cnn_embedding_dim,
                 dropout, Debugging, patch_method, residual=1, is_mask=0, norm_type='LN'):

        super().__init__()

        self.Debugging = Debugging
        self.scale_num = len(Patch_list)
        self.patch_method = patch_method
        self.multi_patch_transformer = nn.ModuleList()

        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = lambda x: x

        if norm_type == 'BN':
            self.norm = nn.BatchNorm2d(cnn_embedding_dim)
        else:
            self.norm = nn.LayerNorm([cnn_encoding_h, cnn_encoding_w])

        self.dropout = nn.Dropout(dropout)

        sub_dim = cnn_embedding_dim // self.scale_num

        for i in range(self.scale_num):
            patch_size_w = Patch_list[i][0]
            patch_size_h = Patch_list[i][1]

            patch_transformer = Patch_Transformer(
                length=length,
                encoding_w=cnn_encoding_w,
                encoding_h=cnn_encoding_h,
                encoding_dim=cnn_encoding_dim,
                patch_size_w=patch_size_w,
                patch_size_h=patch_size_h,
                sub_embedding_dim=sub_dim,
                is_mask=is_mask,
                PATCH_METHOD=patch_method,
                Debugging=Debugging,
            )

            self.multi_patch_transformer.append(patch_transformer)

        self.ffn = Conv_block(cnn_embedding_dim, cnn_embedding_dim, kernel_size=3, stride=1, dilation=1, residual=True)

    def forward(self, c, q, mask):

        x = q
        att = self.residual(x) + self.multi_patch_forward(c, q, mask)
        att = self.norm(att)

        ff = self.residual(att) + self.ffn(att)
        ff = self.norm(ff)

        return ff

    def multi_patch_forward(self, c, q, mask):
        output = []
        for i in range(self.scale_num):
            z = self.multi_patch_transformer[i](c, q, mask)
            output.append(z)
        output = torch.cat(output, 1)  # (6,256,50,50)

        return output


class Prediction_Model(nn.Module):

    def __init__(self, mcof, Length, Width, Height, Input_dim, Patch_list, Encoding_dim, Embedding_dim,
                 Dropout=0.2, Att_num=1, Cross_att_num=1, Is_reduce=1, Is_trans=1, Using_skip=0, Debugging=0, Is_mask=0,
                 residual=1,  Merge='addition', Norm_type='LN', **arg):

        super().__init__()

        self.mcof = mcof
        self.patch_method = mcof.patch_method
        self.Debugging = Debugging
        self.is_reduce = Is_reduce
        self.cross_att_num = Cross_att_num

        self.input_channels = Input_dim
        self.output_channels = Input_dim

        self.merge_type = Merge

        encoding_w = Width
        encoding_h = Height

        self.encoding_dim = Encoding_dim  # 256
        self.embedding_dim = Embedding_dim  # 256

        if self.is_reduce:
            self.dim_factor = 1
        else:
            self.dim_factor = 2

        # 返回的是带有位置编码信息的特征矩阵
        self.loc_pos_enc = LocPositionalEncoder(32 * self.dim_factor, 0.3)  # (32,*,128)
        self.spa_pos_enc = LocPositionalEncoder((Encoding_dim-32)//2 * self.dim_factor, 0.3)

        self.norm_bn = nn.BatchNorm2d(Input_dim)

        self.encoder = Encoder(self.input_channels, self.encoding_dim)
        self.encoder_c = Encoder(self.input_channels, self.encoding_dim)
        self.encoder_q = Encoder(self.input_channels, self.encoding_dim)

        if self.is_reduce:
            tr_encoding_dim = self.encoding_dim
            tr_embedding_dim = self.embedding_dim
        else:
            tr_encoding_dim = self.encoding_dim * 2
            tr_embedding_dim = self.embedding_dim * 2

        if self.is_reduce:
            self.tran0 = Conv_block(self.encoding_dim * 2, self.encoding_dim, kernel_size=1, stride=1, dilation=1, residual=False)
        else:
            self.tran0 = Conv_block(self.encoding_dim, self.encoding_dim * 2, kernel_size=1, stride=1, dilation=1, residual=False)

        if self.is_reduce:
            self.decoder = DecoderRe(self.output_channels, self.encoding_dim, Using_skip, Is_trans, 'Tanh')
        else:
            self.decoder = Decoder(self.output_channels, self.encoding_dim * 2, Using_skip, 'Tanh')

        self.attention_c = nn.ModuleList()
        for a in range(Att_num):
            self.attention_c.append(
                Multi_patch_transfomer(Patch_list, Length, encoding_w, encoding_h, tr_encoding_dim,
                                       tr_embedding_dim, Dropout, Debugging, self.patch_method, residual,
                                       is_mask=Is_mask, norm_type=Norm_type))

        self.attention_cr = nn.ModuleList()
        for a in range(Cross_att_num):
            self.attention_cr.append(
                Multi_patch_transfomer(Patch_list, Length, encoding_w, encoding_h, tr_encoding_dim,
                                       tr_embedding_dim, Dropout, Debugging, self.patch_method, residual,
                                       is_mask=Is_mask, norm_type=Norm_type))

        self.dropout = nn.Dropout(p=Dropout)

        self.linear_tim = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_tim = nn.Sequential(
            nn.Linear(Length * 2 * Height * Width, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2))

        self.linear_typ = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_typ = nn.Sequential(
            nn.Linear(Length * 2 * Height * Width, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3))

        self.feedforward_query = nn.Sequential(
            nn.Linear(Length * 2 * Height * Width, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3))


    def forward(self, avg, que, con):

        # B,T,C,H,W -> BT,C,H,W
        B, T, C, H, W = avg.shape  # (6, 6, 2, 32, 32)

        x_a = avg.reshape(-1, self.input_channels, H, W)  # (B*T, 2, 32, 32)
        x_q = que.reshape(-1, self.input_channels, H, W)  # (B*T, 2, 32, 32)
        x_c = con.reshape(-1, self.input_channels, H, W)

        x_a = self.norm_bn(x_a)
        x_q = self.norm_bn(x_q)
        x_c = self.norm_bn(x_c)

        enc, c3, c2, c1 = self.encoder(x_a)  # (B×T, 256, 32, 32)
        enc_q, c3_q, c2_q, c1_q = self.encoder_q(x_q)  # (B×T, 256, 32, 32)
        enc_c, c3_c, c2_c, c1_c = self.encoder_c(x_c)

        enc = self.dropout(enc)
        enc_q = self.dropout(enc_q)
        enc_c = self.dropout(enc_c)

        # 分类预测
        tim_cls_out = self.tim_class_pred(enc, avg)
        typ_cls_out = self.typ_class_pred(enc, avg)

        # 位置编码
        seq_pos, spa_pos = self.pos_embedding(avg)
        pos = torch.cat((spa_pos, seq_pos), dim=1)
        att_c = torch.cat((enc_c, enc), dim=1)

        if self.is_reduce:
            att_c = self.tran0(att_c)
        else:
            att_c = att_c

        att_c = att_c + pos

        if 1:
            for att_layer in self.attention_c:
                att_c = att_layer(att_c, att_c, None)
            ffn = att_c

        if self.cross_att_num:
            if self.is_reduce:
                att_q = enc_q
            else:
                att_q = self.tran0(enc_q)

            for att_layer in self.attention_cr:
                att_q = att_layer(att_c, att_q, None)
            ffn = att_q

        # [12, 768, 32, 32]
        dec = self.decoder([ffn, c3, c2, c1])

        out = dec.reshape(-1, T, self.output_channels, H, W)

        # que_cls_out = self.query_class_pred(out)

        out = out + avg

        return out, tim_cls_out, typ_cls_out  #, que_cls_out

    def pos_embedding(self, inp):

        B, T, C, H, W = inp.shape
        # (1,T,32) # [B, T, 32, 32, 32]
        pos_t = self.loc_pos_enc(T).permute(1, 2, 0).unsqueeze(-1).type_as(inp)
        pos_t = pos_t.repeat(B, 1, 1, H, W).reshape(B * T, 32 * self.dim_factor, H, W)

        # H位置
        # (1,H,112)->(112,H,1)  # [B, T, 112, 32, 32]
        spa_h = self.spa_pos_enc(H).permute(2, 1, 0).type_as(inp)
        spa_h = spa_h.repeat(B, T, 1, 1, W).reshape(B * T, (self.encoding_dim-32)//2 * self.dim_factor, H, W)

        # W位置
        # (1,W,112)->(112,1,W)  # [B, T, 112, 32, 32]
        spa_w = self.spa_pos_enc(W).permute(2, 0, 1).type_as(inp)
        spa_w = spa_w.repeat(B, T, 1, H, 1).reshape(B * T, (self.encoding_dim-32)//2 * self.dim_factor, H, W)

        spa = torch.cat([spa_h, spa_w], dim=1)

        return pos_t, spa

    def tim_class_pred(self, enc, inp):
        B, T, C, H, W = inp.shape

        enc = self.linear_tim(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_tim(enc)

        # cls_out = self.softmax_tim(enc)

        return enc

    def typ_class_pred(self, enc, inp):
        B, T, C, H, W = inp.shape

        enc = self.linear_typ(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_typ(enc)

        return enc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--mode', type=str, default='train', help='The processing phase of the model')
    parser.add_argument('--record', type=str, help='Recode ID')
    parser.add_argument('--task', type=str, default='B', help='Processing task type')
    parser.add_argument('--keep_train', type=int, default=0, help='Model keep training')
    parser.add_argument('--epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--inp_type', type=str, default='external',
                        choices=['external', 'accumulate', 'accumulate_avg', 'train', 'holiday', 'windspeed', 'weather',
                                 'temperature'])
    parser.add_argument('--patch_method', type=str, default='UNFOLD', choices=['EINOPS', 'UNFOLD', 'STTN'])

    parser.add_argument('--pos_en', type=int, default=1, help='positional encoding')
    parser.add_argument('--pos_en_mode', type=str, default='cat', help='positional encoding mode')
    mcof = parser.parse_args()

    PATCH_LIST = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 8], [8, 8], [8, 8], [8, 8]]
    net = Prediction_Model(
        mcof=mcof,
        Length=6,  # 8
        Width=8,  # 200
        Height=16,  # 200
        Input_dim=2,  # 1
        Patch_list=PATCH_LIST,  # 小片段的大小
        Att_num=1,  # 2
        Cross_att_num=1,  # 2
        Using_skip=2,  # 1
        Encoding_dim=256,  # 256
        Embedding_dim=256,  # 256
        Is_mask=1,  # 1
        Is_reduce=1,
        Debugging=0,  # 0
        Merge='cross-attention',  # cross-attention
        Norm_type='LN'
    )

    input_c = torch.randn(2, 6, 2, 16, 8)
    input_q = torch.randn(2, 6, 2, 16, 8)
    context_c = torch.randn(2, 6, 2, 16, 8)


    out, tim_out, typ_out = net(input_c, input_q, context_c)
    print('=============')
    print(out.shape, tim_out.shape)
    summary(net, input_c, input_q, context_c)
