import torch
import numpy as np
import random

####DATA#####

AM_NUM = 1440


def data_set(TRAIN_DATESET, INITIALIZATION, IS_FIT, INTERVAL=5, ADJUST_MAX=1.0, SEQ_LEN=4061):
    if SEQ_LEN == 4061:
        MAIN_FOLDER = 'data_niu'
        BASE_COUNT = 240
    else:
        MAIN_FOLDER = 'data_{}'.format(SEQ_LEN)
        BASE_COUNT = 12

    total_list = []
    for TRAIN_DATE in TRAIN_DATESET:

        if IS_FIT:
            npy_path = 'data_niu_C/{}_norm_ALL_seq{}_niu.npy'.format(TRAIN_DATE, INTERVAL)
        else:
            npy_path = '{}/{}_norm_ALL_seq{}_niu.npy'.format(MAIN_FOLDER, TRAIN_DATE, INTERVAL)

        data_np = np.load(npy_path)
        data_np = np.transpose(data_np, (0, 1, 4, 2, 3))
        data_pth = torch.from_numpy(data_np).float()
        data_pth = data_pth * ADJUST_MAX

        if TRAIN_DATE == '2020-01-24':
            data_pth = data_pth[:AM_NUM]

        total_list.append(data_pth)
        print(npy_path, data_pth.shape)

    total_pth = torch.cat(total_list, 0)

    seq = total_pth
    MEAN = seq.mean()
    print('TRAIN_SET:', 'shape:', seq.shape, 'MAX:', torch.max(seq), 'MEAN:', MEAN)
    seq_label = seq

    if INITIALIZATION == 'avg_hole':

        DATE_TARGET = '2020-01-24'
        DATE_AVG_LIST = [
            '2020-01-17',
            '2020-01-18',
            '2020-01-19',
            '2020-01-20',
            '2020-01-21',
            # '2020-01-23',
            # '2020-01-24',
        ]
        seq_hole, _, _ = val_data_prediction(DATE_TARGET, DATE_AVG_LIST, INTERVAL, SEQ_LEN, MODE='ALL')

        N, T, C, H, W = seq_hole.shape
        seq_hole = seq_hole.expand(len(total_list), N, T, C, H, W)
        seq_hole = seq_hole.reshape(len(total_list) * N, T, C, H, W)
    else:
        seq_hole = seq

    return seq, seq_label, MEAN, seq_hole


def val_data(EVAL_DATE, INITIALIZATION, INTERVAL, ADJUST_MAX, SEQ_LEN=4061, MODE='ALL'):
    if SEQ_LEN == 4061:
        MAIN_FOLDER = 'data_niu'
    else:
        MAIN_FOLDER = 'data_NYC_{}'.format(SEQ_LEN)

    DATE = EVAL_DATE

    npy_path = '{}/{}_norm_{}_seq{}_nycinp.npy'.format(MAIN_FOLDER, DATE, MODE, INTERVAL)

    data_np = np.load(npy_path)
    data_np = np.transpose(data_np, (0, 1, 4, 2, 3))
    data_pth = torch.from_numpy(data_np).float()
    data_pth = data_pth * ADJUST_MAX
    seq = data_pth
    MEAN = seq.mean()
    print('EVAL_SET:', DATE, 'shape:', seq.shape, 'MAX:', torch.max(seq), 'MEAN:', MEAN)
    _seq_label = seq
    _seq = seq

    # DATE = '2019-07-19'
    # npy_path = 'data_xie/{}_norm_ALL_seq{}_xie.npy'.format(DATE,INTERVAL)

    # data_np = np.load(npy_path)
    # data_np = np.transpose(data_np,(0,1,4,2,3))
    # data_pth = torch.from_numpy(data_np).float()
    # seq = data_pth
    # MEAN = seq.mean()
    # print('EVAL_SET:',DATE,'shape:',seq.shape,'MAX:',torch.max(seq),'MEAN:',MEAN)
    # _seq_label = seq

    if INITIALIZATION == 'avg_hole':

        DATE_TARGET = '2014-04-16'

        DATE_AVG_LIST = [
            '2014-04-11',
            '2014-04-12',
            '2014-04-13',
            '2014-04-14',
            '2014-04-15',
        ]

        seq_hole, _, _ = val_data_prediction(DATE_TARGET, DATE_AVG_LIST, INTERVAL, SEQ_LEN, MODE=MODE)
    else:
        seq_hole = _seq

    return _seq, _seq_label, MEAN, seq_hole


def data_set_prediction(TRAIN_CONTEXT_DATE, TRAIN_AVG_DATE, INTERVAL=5, RANDOM_CONTEXT=0, ADJUST_MAX=1.0, SEQ_LEN=4061):
    LEN = INTERVAL
    # BASE_LEN = 6
    BASE_LEN = 6
    MUL = LEN // BASE_LEN

    TOTAL_TARGET_LIST = []
    TOTAL_CONTEXT_LIST = []

    # if SEQ_LEN == 4061:
    #     MAIN_FOLDER = 'data_niu'
    #     BASE_COUNT = 240
    # else:
    # MAIN_FOLDER = 'data_xie_{}'.format(SEQ_LEN)
    MAIN_FOLDER = 'data_BJ_{}'.format(SEQ_LEN)
    BASE_COUNT = 4

    for DATE_TARGET in TRAIN_CONTEXT_DATE:

        if RANDOM_CONTEXT == 1:

            # if DATE_TARGET == '2019-07-22':
            # AM = np.load('{}_B/{}_norm_AM_xie.npy'.format(MAIN_FOLDER,DATE_TARGET,LEN,LEN))
            # else:
            # AM = np.load('{}/{}_norm_AM_xie.npy'.format(MAIN_FOLDER,DATE_TARGET,LEN,LEN))

            AM = np.load('{}/{}_norm_AM_bjinp.npy'.format(MAIN_FOLDER, DATE_TARGET, LEN, LEN))

            CONTEXT_TIMES = 6 * MUL                 # 6
            # CONTEXT_TIMES = 3 * MUL                 # 3
            CONTEXT_COUNT = BASE_COUNT // MUL       # 4

            LEN = INTERVAL

            CONTEXT_LIST = []

            for off in range(CONTEXT_COUNT):
                context_list = []
                for i in range(CONTEXT_TIMES):
                    context_list.append(AM[i * CONTEXT_COUNT + off])
                context = np.stack(context_list, 0)     # (6,32,32,1)
                context = np.transpose(context, (0, 3, 1, 2))
                context = torch.from_numpy(context).float()
                context = context * ADJUST_MAX
                CONTEXT_LIST.append(context)   # (4,6,32,32,1)

        # if DATE_TARGET == '2019-07-22':
        # data_target = np.load('{}_B/{}_norm_ALL_seq{}_xie.npy'.format(MAIN_FOLDER,DATE_TARGET,LEN,LEN))
        # else:
        # data_target = np.load('{}/{}_norm_ALL_seq{}_xie.npy'.format(MAIN_FOLDER,DATE_TARGET,LEN,LEN))
        data_target = np.load('{}/{}_norm_ALL_seq{}_bjinp.npy'.format(MAIN_FOLDER, DATE_TARGET, LEN, LEN))
        data_target = np.transpose(data_target, (0, 1, 4, 2, 3))
        data_target = torch.from_numpy(data_target).float()
        data_target = data_target * ADJUST_MAX
        print('target', DATE_TARGET, data_target.shape)

        context_seq = []
        for i in range(len(data_target)):      # 22
            idx = random.randint(0, len(CONTEXT_LIST) - 1)
            context = CONTEXT_LIST[idx]        # 4
            context_seq.append(context)
        context = torch.stack(context_seq, 0)     # 22  using AM data random stack
        print('target_context', DATE_TARGET, context.shape)

        TOTAL_TARGET_LIST.append(data_target)
        TOTAL_CONTEXT_LIST.append(context)

    total_target = torch.stack(TOTAL_TARGET_LIST, 0)     # [B,N,T,C,H,W]->[7,22,3,1,200,200]
    print(total_target.shape)
    total_context = torch.stack(TOTAL_CONTEXT_LIST, 0)
    B, N, T, C, H, W = total_target.shape
    total_target = total_target.reshape(B * N, T, C, H, W)    # [154,3,1,200,200]
    total_context = total_context.reshape(B * N, T, C, H, W)
    print('total_target', total_target.shape)

    DATA_LIST = []
    for DATE_INPUT in TRAIN_AVG_DATE:
        # if DATE_INPUT == '2019-07-22' or DATE_INPUT == '2019-07-01' or DATE_INPUT == '2019-07-02':
        # data_input = np.load('{}_B/{}_norm_ALL_seq{}_xie.npy'.format(MAIN_FOLDER,DATE_INPUT,LEN))
        # else:
        # data_input = np.load('{}/{}_norm_ALL_seq{}_xie.npy'.format(MAIN_FOLDER,DATE_INPUT,LEN))
        data_input = np.load('{}/{}_norm_ALL_seq{}_bjinp.npy'.format(MAIN_FOLDER, DATE_INPUT, LEN))
        data_input = np.transpose(data_input, (0, 1, 4, 2, 3))
        data_input = torch.from_numpy(data_input).float()
        data_input = data_input * ADJUST_MAX
        print('input', DATE_INPUT, data_input.shape)
        DATA_LIST.append(data_input)

    AVERAGE = DATA_LIST[0]
    if len(DATA_LIST) > 1:
        for i in range(1, len(DATA_LIST)):
            AVERAGE = AVERAGE + DATA_LIST[i]      # using add strategy to obtain data input 17\18\19\20\21

    AVERAGE = AVERAGE / len(DATA_LIST)
    data_input = AVERAGE
    print('avg_input', data_input.shape)

    TOTAL_INPUT_LIST = []
    for i in range(len(TRAIN_CONTEXT_DATE)):
        TOTAL_INPUT_LIST.append(data_input)

    total_input = torch.stack(TOTAL_INPUT_LIST, 0)
    B, N, T, C, H, W = total_input.shape
    total_input = total_input.reshape(B * N, T, C, H, W)
    print('total_input', total_input.shape)

    return total_input, total_target, total_context


def val_data_prediction(DATE_TARGET, DATE_AVG_INPUT_LIST, INTERVAL, ADJUST_MAX, SEQ_LEN, MODE='PM'):
    LEN = INTERVAL
    # if SEQ_LEN == 4061:
    #     MAIN_FOLDER = 'data_niu'
    #     BASE_COUNT = 240
    # else:
    MAIN_FOLDER = 'data_BJ_{}'.format(SEQ_LEN)
    # BASE_COUNT = 12

    context_target = np.load('{}/{}_norm_CONTEXT_L{}_bjinp.npy'.format(MAIN_FOLDER, DATE_TARGET, LEN))
    context_target = np.transpose(context_target, (0, 3, 1, 2))
    context_target = torch.from_numpy(context_target).float()
    context_target = context_target * ADJUST_MAX
    print(DATE_TARGET, 'context', context_target.shape)

    data_target = np.load('{}/{}_norm_{}_seq{}_bjinp.npy'.format(MAIN_FOLDER, DATE_TARGET, MODE, LEN))
    data_target = np.transpose(data_target, (0, 1, 4, 2, 3))
    data_target = torch.from_numpy(data_target).float()
    data_target = data_target * ADJUST_MAX
    print(DATE_TARGET, 'target', data_target.shape)

    DATA_LIST = []
    for DATE_INPUT in DATE_AVG_INPUT_LIST:
        # if DATE_INPUT == '2019-07-22' or DATE_INPUT == '2019-07-01' or DATE_INPUT == '2019-07-02':
        if DATE_INPUT == '2020-01-30' or DATE_INPUT == '2020-01-31':
            data_input = np.load('{}_B/{}_norm_{}_seq{}_bjinp.npy'.format(MAIN_FOLDER, DATE_INPUT, MODE, LEN))
        else:
            data_input = np.load('{}/{}_norm_{}_seq{}_bjinp.npy'.format(MAIN_FOLDER, DATE_INPUT, MODE, LEN))

        data_input = np.transpose(data_input, (0, 1, 4, 2, 3))
        data_input = torch.from_numpy(data_input).float()
        data_input = data_input * ADJUST_MAX
        print('--', DATE_INPUT, 'input', data_input.shape, 'MAX', data_input.max(), 'MEAN', data_input.mean())
        DATA_LIST.append(data_input)

    AVERAGE = DATA_LIST[0]
    if len(DATA_LIST) > 1:
        for i in range(1, len(DATA_LIST)):
            AVERAGE = AVERAGE + DATA_LIST[i]

    AVERAGE = AVERAGE / len(DATA_LIST)
    data_input = AVERAGE

    print('--', 'average input', data_input.shape, 'MAX', data_input.max(), 'MEAN', data_input.mean())

    return data_input, data_target, context_target


if __name__ == '__main__':
    DATASET = 'bj1'
    seq, seq_label, MEAN = data_set(DATASET, INTERVAL=3)
    print(seq.shape)
