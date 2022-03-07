import torch
from torch import nn
from util.util import timeSince, get_yaml_data, getting_mask, initialize_hole, loadding_mask
from util.data import data_set, val_data, val_data_prediction
import numpy as np
import cv2
from tqdm import tqdm

# ABSOLUTE_ZEROS_PATH = 'data/all_mask_64.png'
# KIOSK_PATH = 'data/kiosk.png'
ZEROS_THRESHOLD = 0.0045
ONES_THRESHOLD = 1.0
# AM_NUM = 1440
# AM_NUM_24 = 24
AM_NUM = 12
# AFTER_KIOSK_NUM_132 = 84
MASK_PATH = 'data/hole_64.png'


def validate(net, EVAL_DATE, INTERVAL, mask_path, INITIALIZATION, C, SEQ_LEN, model_path='model/model.pth',
             ADJUST_MAX=1.0, IS_MERGE=0, IS_4061=0, MODE='ALL', DATA=None):
    net.load_state_dict(torch.load(model_path))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # net.load_state_dict(torch.load(model_path,map_location=device))
    net = net.to(device)
    net = nn.DataParallel(net)
    # model_path = 'model/model.pth'
    net.eval()

    if DATA == None:
        seq, seq_label, MEAN, seq_hole = val_data(EVAL_DATE, INITIALIZATION, INTERVAL, ADJUST_MAX, SEQ_LEN)
    else:
        seq, seq_label, MEAN, seq_hole = DATA

    ###Five Minutes

    if MODE == 'ALL':
        five_T = 24
    if MODE == 'PM':
        five_T = 12
    if MODE == 'AM':
        five_T = 12

    if SEQ_LEN == 4061:
        INTER = 20
        seq_24_out = []
        seq_24_lab = []
        seq_24_hol = []
        for k in range(five_T - 1):
            seq_24_out.append(seq[k * INTER])
            seq_24_lab.append(seq_label[k * INTER])
            seq_24_hol.append(seq_hole[k * INTER])

        seq_24_out.append(seq[-1])
        seq_24_lab.append(seq_label[-1])
        seq_24_hol.append(seq_hole[-1])

        seq = torch.stack(seq_24_out, 0)
        seq_label = torch.stack(seq_24_lab, 0)
        seq_hole = torch.stack(seq_24_hol, 0)
    elif SEQ_LEN == 24:
        INTER = 1

    N = seq.shape[0]
    interval = seq.shape[1]

    out_list = []
    label_list = []
    input_list = []
    av_hole_list = []

    for n in (range(N)):

        input = seq[n:n + 1]
        label = seq_label[n:n + 1]
        avg_hole = seq_hole[n:n + 1]
        hole, valid, single_valid = loadding_mask(input, mask_path, (96, 64))
        input = initialize_hole(INITIALIZATION, C, MEAN, input, hole, valid, avg_hole)
        input = input.to(device)
        label = label.to(device)
        single_valid = single_valid.to(device)
        output = net(input, single_valid)

        b = 0
        t = 0
        if n == (N - 1): t = -1

        out_list.append(output[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
        label_list.append(label[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
        input_list.append(input[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
        av_hole_list.append(avg_hole[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy())

    output = np.stack(out_list, 0)
    label = np.stack(label_list, 0)
    input = np.stack(input_list, 0)
    av_hole = np.stack(av_hole_list, 0)

    # RULE
    new_o = []
    for f in (range(output.shape[0])):
        o = output[f]
        o[o < ZEROS_THRESHOLD] = 0.0
        new_o.append(o)

    output = np.stack(new_o, 0)

    output = output * 1.0

    # CONSTANT = 0.65
    # output = output*CONSTANT

    ###Score
    valid = cv2.imread(mask_path)
    valid = cv2.cvtColor(valid, cv2.COLOR_BGR2GRAY)

    valid_num = valid > 0
    valid_num = valid_num.sum()
    valid_total = valid_num * output.shape[0]

    score_list = []

    for f in (range(output.shape[0])):
        valid = (valid > 0)

        o = output[f]
        t = label[f]
        i = input[f]
        av_h = av_hole[f]

        h, w, c = o.shape

        o = o.reshape(h, w)
        t = t.reshape(h, w)
        i = i.reshape(h, w)
        av_h = av_h.reshape(h, w)

        # ALL_ZEROS
        # o = np.zeros([h,w])
        # OTHER_DAYS
        # o = i
        # AV_H
        # o = av_h

        o = o * valid

        t = t * valid
        l_t = t * 0.5
        h_t = t * 2.0

        con1 = (o >= l_t)
        con2 = (o <= h_t)

        score = con1 * con2
        score = score * valid
        score = score.sum()
        score_list.append(score)

    sum_score = sum(score_list)

    final_score = sum_score / valid_total

    if IS_MERGE:
        ##MASKING
        T, H, W, C = output.shape
        TOTAL_MASK = cv2.imread('data/hole_64.png')
        TOTAL_MASK = cv2.cvtColor(TOTAL_MASK, cv2.COLOR_BGR2GRAY) // 255
        TOTAL_MASK = TOTAL_MASK.reshape(H, W, C)
        TOTAL_MASK = np.tile(TOTAL_MASK, (T, 1, 1, 1))

        output = output * TOTAL_MASK + label * (1 - TOTAL_MASK)

    output_4061 = None
    return final_score, sum_score, valid_total, output, label, input, output_4061


def validate_pre(net, INTERVAL, C, ADJUST_MAX, SEQ_LEN,
                 model_path='model/pre_model.pth', IS_MERGE=0, DATA=None):
    # net.load_state_dict(torch.load(model_path))
    net.load_state_dict(torch.load(model_path))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    net = nn.DataParallel(net)

    net.eval()

    if DATA == None:
        pass
        # seq_last, seq_label, context = val_data_prediction(DATE_TARGET, DATE_AVG_INPUT_LIST, INTERVAL, ADJUST_MAX, SEQ_LEN,
        #                                              MODE)
    else:
        # X, X_ave, _, Y, _ = data
        context, last, _, label, _ = DATA
        context = np.expand_dims(context, axis=2)
        seq_last = np.expand_dims(last, axis=2)
        seq_label = np.expand_dims(label, axis=2)
        # B, T, C, H, W = context.shape
        # last, label, context = data
        print('查看评估数据集：输入：{0}，目标：{1}，上下文：{2}'.format(seq_last.shape, seq_label.shape, context.shape))

    N = seq_last.shape[0]    # 10
    interval = seq_last.shape[1]

    out_list = []
    last_list = []
    label_list = []
    input_list = []

    for n in (range(N)):

        # torch.Size([1, 3, 1, 200, 200]) torch.Size([1, 3, 1, 200, 200]) torch.Size([1, 3, 1, 200, 200])
        last = seq_last[n:n + 1]
        label = seq_label[n:n + 1]
        context = context.expand_as(last)

        last = last.to(device)
        label = label.to(device)
        context = context.to(device)

        output = net(last, context)

        b = 0
        t = 0
        if n == (N - 1): t = -1

        # 当前帧
        las = last[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        out = output[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
        lab = label[b, t, :, :, :].detach().cpu().permute(1, 2, 0).numpy()  # 1 frame

        last_list.append(las)
        out_list.append(out)
        label_list.append(lab)

    last = np.stack(last_list, 0)
    output = np.stack(out_list, 0)
    label = np.stack(label_list, 0)

    # scale to original value
    # MAX = 1266 #2019.07.19

    ###Score
    # valid = cv2.imread(MASK_PATH)
    # valid = cv2.cvtColor(valid, cv2.COLOR_BGR2GRAY)
    #
    # valid_num = valid > 0
    # valid_num = valid_num.sum()
    # valid_total = valid_num * output.shape[0]
    valid_total = output.shape[0]   # 10

    ###Score
    # valid_total = output.shape[0] * output.shape[1] * output.shape[2]

    score_list = []

    ZEROS_THRESHOLD = 0.0045
    # ZEROS_THRESHOLD = 0.0008*9

    # RULE
    new_o = []
    for f in (range(output.shape[0])):
        o = output[f]
        o[o < ZEROS_THRESHOLD] = 0.0
        new_o.append(o)

    output = np.stack(new_o, 0)

    output = output * 1.0

    for f in (range(output.shape[0])):
        # valid = (valid > 0)

        o = output[f]
        t = label[f]
        l = last[f]

        # ZEROS
        # o = np.zeros_like(o)

        # LAST
        # o = l

        # TARGET
        # o = t

        ##MERGING
        # o = 0.5*o + 0.5*l

        h, w, c = o.shape

        o = o.reshape(h, w)
        t = t.reshape(h, w)

        # o = o * valid
        # t = t * valid

        l_t = t * 0.5
        h_t = t * 2.0

        con1 = (o >= l_t)
        con2 = (o <= h_t)

        score = con1 * con2
        # score = score * valid
        score = score.sum()

        score_list.append(score)

    sum_score = sum(score_list)

    final_score = sum_score / valid_total

    if IS_MERGE:
        output[:AM_NUM] = label[:AM_NUM]

    return final_score, sum_score, valid_total, output, label, last
