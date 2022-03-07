import time
import math
import yaml
import torch
import cv2
import pandas as pd
import datetime
import tqdm
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def random_mask(H, W):
    t = 30

    left = random.randint(0, W - t)
    top = random.randint(0, H - t)

    right = random.randint(left + t, W)
    bottom = random.randint(top + t, H)

    return (top, bottom, left, right)


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()

    # data = yaml.load(file_data,Loader=yaml.Fullloader)
    data = yaml.load(file_data)
    return data


def loadding_mask(input, path, size=(160, 120)):
    # [64, 5, 1, 120, 160]

    mask = cv2.imread(path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = cv2.resize(mask,(160,120))
    mask = cv2.resize(mask, size)
    mask = torch.from_numpy(mask)
    h, w = mask.shape
    mask = mask // 255
    mask = mask.reshape(1, h, w)
    mask = mask.expand_as(input)
    mask = mask.to(input.device)
    mask = mask.float()
    print(mask[:, :, :, 60:80, 90:100])

    valid = mask
    hole = 1 - mask
    single_valid = mask
    # single_valid = single_valid.to(input.device)

    return hole, valid, single_valid


def getting_mask(input, mask_t, mask_b, mask_l, mask_r, is_random=False):
    N, T, C, H, W = input.shape

    if is_random:
        mask_t, mask_b, mask_l, mask_r = random_mask(H, W)

    mask = torch.zeros_like(input)
    mask[:, :, :, mask_t:mask_b, mask_l:mask_r] = 1

    valid = mask
    hole = 1 - mask

    single_valid = torch.zeros(N, T, 1, H, W)
    single_valid[:, :, :, mask_t:mask_b, mask_l:mask_r] = 1
    single_valid = single_valid.to(input.device)

    return hole, valid, single_valid


def initialize_hole(INITIALIZATION, C, MEAN, seq, hole, valid, avg_hole):
    if INITIALIZATION == 'mean':

        if C == 3:
            valid = valid.transpose(1, 2)
            valid[:, 0, :, :, :] = valid[:, 0, :, :, :] * MEAN[0]
            valid[:, 1, :, :, :] = valid[:, 1, :, :, :] * MEAN[1]
            valid[:, 2, :, :, :] = valid[:, 2, :, :, :] * MEAN[2]
            valid = valid.transpose(1, 2)
        elif C == 1:
            valid = valid * MEAN

        seq = seq * hole + valid

    elif INITIALIZATION == 'avg_hole':
        seq = seq * hole + valid * avg_hole
        # seq = valid

    elif INITIALIZATION == 'white':

        seq = seq * hole + valid

    elif INITIALIZATION == 'black':

        seq = seq * hole

    return seq


def cv_show(img, H, W, C, name='img'):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)
    cv2.imshow(name, img)
    cv2.waitKey(3000)


def cat_input_valid(input, single_valid):
    input = input.transpose(1, 2)
    single_valid = single_valid.transpose(1, 2)
    input = torch.cat([input, single_valid], dim=1)
    input = input.transpose(1, 2)

    img = input[0, 0, 0:3, :, :]
    img = img.cpu().detach().numpy()
    H = 64
    W = 96
    cv_show(img, H=H, W=W, C=3)

    return input


# with open('../data/HOLE_MESH_LIST.pkl', 'rb') as f:
#     HOLE_MESH_LIST = pickle.load(f)

AM_COUNT = 24
PM_COUNT = 48 - AM_COUNT


def output_csv(output, DATE, csv_file, NORMAL_MAX, MODE='HOLE'):
    if MODE == 'PM':
        START_TIME = '{} 12:00:00'.format(DATE)
        if len(output) == 48:
            output = output[AM_COUNT:]
        if len(output) == PM_COUNT:
            pass

    else:
        START_TIME = '{} 00:00:00'.format(DATE)

    START_TIME = datetime.datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")

    T, H, W, C = output.shape
    # (10,8,16,1)-->(10,8,16)
    output = output.reshape((T, H, W))
    # 24,200,200,1

    # MAX = 1291 #2019-07-19 csv
    MAX = NORMAL_MAX

    # VALID_H = 32
    # VALID_W = 32
    VALID_H = 32
    VALID_W = 32

    data_list = []
    cnt = 0
    hole_cnt = 0
    for t in (range(T)):
        for h in range(VALID_H):
            for w in range(VALID_W):
                cnt += 1
                count = output[t][h][w] * MAX
                count = round(count, 4)

                count = int(count)
                if count < 1.0: count = 0

                ori_x = w
                ori_y = h
                meshname = '{},{}'.format(ori_x, ori_y)
                TIME = (START_TIME + datetime.timedelta(minutes=30 * t)).strftime("%Y-%m-%d %H:%M:%S")
                # if MODE == 'HOLE':
                # if meshname in HOLE_MESH_LIST:
                data_list.append([TIME, meshname, count])
                    # hole_cnt += 1
                # else:
                #    data_list.append([TIME,meshname,count])
                #    hole_cnt += 1
    if MODE == 'PM':
        assert cnt == T * VALID_H * VALID_W
    # if MODE == 'HOLE':
    #     assert hole_cnt == T * len(HOLE_MESH_LIST)

    data = pd.DataFrame(data_list, columns=['datetime', 'meshname', 'count'])
    data.sort_values(by=['datetime', 'meshname'], inplace=True, ascending=[True, True])
    data.to_csv(csv_file, index=False)


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        # nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
        nn.init.xavier_normal_(model.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
        # nn.init.xavier_normal_(model.weight.data)


def VALRMSE(input, target, ds, m_factor):
    # print(input.shape, target.shape)
    # input = torch.tensor(input.data.cpu().numpy() * ds.img_std + ds.img_mean)
    # target = torch.tensor(target.data.cpu().numpy() * ds.img_std + ds.img_mean)
    rmse = torch.sqrt(F.mse_loss(input, target)) * (ds.mmn.max - ds.mmn.min) / 2. * m_factor
    return rmse


def VALMAPE(input, target, mmn, m_factor):
    mape = torch.mean(torch.abs((target - input) / input))
    return mape


if __name__ == '__main__':
    input = torch.zeros([64, 5, 1, 120, 160])
    hole, valid, single_valid = loadding_mask(input, '../data/hole_64.png')
    print(hole.shape, valid.shape, single_valid.shape)
