import numpy as np
import pickle
import time
import sys
import random
import argparse
import os
import math
from time import localtime, strftime
from sklearn import metrics
from einops import rearrange
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
from util.util import timeSince, get_yaml_data
from util.util import VALRMSE, VALMAPE
import shutil

from net.msp_sttn import Prediction_Model as Model
#from net.imp_pos_cl_heat2heat import Prediction_Model as Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

TORCH_VERSION = torch.__version__

seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class DataConfiguration:
    def __init__(self, Len_close, Len_period, Len_trend):
        super().__init__()

        # Data
        self.name = 'BikeNYC'
        self.portion = 1.  # portion of data

        self.len_close = Len_close
        self.len_period = Len_period
        self.len_trend = Len_trend
        self.pad_forward_period = 0
        self.pad_back_period = 0
        self.pad_forward_trend = 0
        self.pad_back_trend = 0

        self.len_all_close = self.len_close * 1
        self.len_all_period = self.len_period * (1 + self.pad_back_period + self.pad_forward_period)
        self.len_all_trend = self.len_trend * (1 + self.pad_back_trend + self.pad_forward_trend)

        self.len_seq = self.len_all_close + self.len_all_period + self.len_all_trend
        self.cpt = [self.len_all_close, self.len_all_period, self.len_all_trend]

        self.interval_period = 1
        self.interval_trend = 7

        self.ext_flag = True
        self.ext_time_flag = True
        self.rm_incomplete_flag = True
        self.fourty_eight = True
        self.previous_meteorol = True

        self.dim_h = 16
        self.dim_w = 8


def run(mcof):
    IS_TRAIN = 0
    IS_VAL = 0
    ####SETTING####
    INP_TYPE = mcof.inp_type
    DATA_TYPE = mcof.dataset_type
    RECORD_ID = mcof.record
    PRESUME_RECORD_ID = mcof.presume_record
    EPOCH_S = mcof.epoch_s
    PRESUME_EPOCH_S = mcof.presume_epoch_s
    IS_BEST = mcof.best
    IS_RECT = mcof.is_rect

    if len(mcof.mode) > 1:
        if mcof.mode == 'train':
            IS_TRAIN = 1
            setting = get_yaml_data("./pre_setting_nyc.yaml")
            BATCH_SIZE = setting['TRAIN']['BATCH_SIZE']
        if mcof.mode == 'val':
            IS_VAL = 1
            BATCH_SIZE = 24
            RECORD_ID = mcof.record
            setting = get_yaml_data(f"./record/{RECORD_ID}/pre_setting_nyc.yaml")

    ####SETTING####
    DROPOUT = setting['TRAIN']['DROPOUT']
    MERGE = setting['TRAIN']['MERGE']
    PATCH_LIST = setting['TRAIN']['PATCH_LIST']
    PATCH_LIST = eval(PATCH_LIST)
    IS_USING_SKIP = setting['TRAIN']['IS_USING_SKIP']
    MODEL_DIM = setting['TRAIN']['MODEL_DIM']
    ATT_NUM = setting['TRAIN']['ATT_NUM']
    CROSS_ATT_NUM = setting['TRAIN']['CROSS_ATT_NUM']
    IS_MASK_ATT = setting['TRAIN']['IS_MASK_ATT']
    LR = setting['TRAIN']['LR']
    EPOCH_E = setting['TRAIN']['EPOCH']
    WARMUP_EPOCH = setting['TRAIN']['WARMUP_EPOCH']
    MILE_STONE = setting['TRAIN']['MILE_STONE']
    LOSS_MAIN = setting['TRAIN']['LOSS_MAIN']
    LOSS_TIM = setting['TRAIN']['LOSS_TIM']
    LOSS_TYP = setting['TRAIN']['LOSS_TYP']
    LEN_CLOSE = setting['TRAIN']['LEN_CLOSE']
    LEN_PERIOD = setting['TRAIN']['LEN_PERIOD']
    LEN_TREND = setting['TRAIN']['LEN_TREND']
    LENGTH = setting['TRAIN']['LENGTH']
    IS_SEQ = setting['TRAIN']['IS_SEQ']
    IS_REDUCE = setting['TRAIN']['IS_REDUCE']
    EVAL_START_EPOCH = setting['TRAIN']['EVAL_START_EPOCH']
    EVAL_BATCH = setting['TRAIN']['EVAL_BATCH']
    SEQ_LEN_TEST = setting['TRAIN']['SEQ_LEN_TEST']
    SEQ_LEN_TRAIN = setting['TRAIN']['SEQ_LEN_TRAIN']

    C = 2
    H = 16
    W = 8

    from dataset.dataset import DatasetFactory

    dconf = DataConfiguration(Len_close=LEN_CLOSE,
                              Len_period=LEN_PERIOD,
                              Len_trend=LEN_TREND,
                              )
    ds_factory = DatasetFactory(dconf, INP_TYPE, DATA_TYPE, LENGTH, IS_SEQ)

    if IS_TRAIN:

        try:
            if os.path.exists('./record/{}/'.format(RECORD_ID)):
                shutil.rmtree('./record/{}/'.format(RECORD_ID))
            os.makedirs('./record/{}/'.format(RECORD_ID))

            oldname = os.getcwd() + os.sep
            newname = f'./record/{RECORD_ID}/'
            shutil.copyfile(oldname + 'pre_setting_nyc.yaml', newname + 'pre_setting_nyc.yaml')
            shutil.copyfile(oldname + 'pre_main_short.py', newname + 'pre_main_short.py')
            shutil.copytree(oldname + 'net', newname + 'net')
            shutil.copytree(oldname + 'dataset', newname + 'dataset')
        except:
            raise print('record directory not find!')

        record = open("record/{}/log.txt".format(RECORD_ID), "w")

        curr_time = strftime('%y%m%d%H%M%S', localtime())
        Keep_Train = mcof.keep_train


        train_ds = ds_factory.get_train_dataset()

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=1
        )

        ####MODEL####


        net = Model(
            mcof, Length=LENGTH, Width=W, Height=H, Input_dim=C,
            Patch_list=PATCH_LIST, Dropout=DROPOUT, Att_num=ATT_NUM,
            Cross_att_num=CROSS_ATT_NUM, Using_skip=IS_USING_SKIP,
            Encoding_dim=MODEL_DIM, Embedding_dim=MODEL_DIM,
            Is_mask=IS_MASK_ATT, Is_reduce=IS_REDUCE,
            Debugging=0,Merge=MERGE,
        )

        ####TRAINING####
        print('TRAINING START')
        print('-' * 30)


        start = time.time()

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device_ids = [i for i in range(torch.cuda.device_count())]

        #### Optimizer ####
        optimizer = optim.Adam(net.parameters(), lr=eval(LR))

        gamma = 0.5
        warm_up_with_multistep_lr = lambda epoch: epoch / int(WARMUP_EPOCH) if epoch <= int(
            WARMUP_EPOCH) else gamma ** len([m for m in eval(MILE_STONE) if m <= epoch])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

        #### Loss Function ####
        # if LOSS == 'L1':
        #     if IS_SOFT_LABEL:
        #         criterion = torch.nn.L1Loss(reduction='none')
        #     else:
        #         criterion = torch.nn.L1Loss()
        # elif LOSS == 'L2':
        #     if IS_SOFT_LABEL:
        #         criterion = torch.nn.MSELoss(reduction='none')
        #     else:
        #         criterion = torch.nn.MSELoss()

        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        class_criterion = nn.CrossEntropyLoss()

        if Keep_Train:
            path = './model/Imp_{}/pre_model_{}.pth'.format(PRESUME_RECORD_ID, PRESUME_EPOCH_S)
            # net.load_state_dict(torch.load(path))
            pretrained_dict = torch.load(path)
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
        else:
            pass

        #### 训练设备准备
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)

        #### Training ####
        it = 0
        for epoch in range(0, EPOCH_E):

            net.train()
            for i, data in enumerate(train_loader):

                con, ave, ave_q, label, tim_cls, typ_cls = data

                B, T, C, H, W = con.shape
                ave = ave.to(device)
                ave_q = ave_q.to(device)
                con = con.to(device)
                label = label.to(device)
                tim_cls = tim_cls.squeeze().to(device)
                typ_cls = typ_cls.squeeze().to(device)

                optimizer.zero_grad()

                out, tim_out, typ_out = net(ave, ave_q, con)

                out = out.reshape(B, T, C, H, W)

                #### 将模型输出进行均值处理 ####

                #train
                if IS_SEQ:
                    oup = out[:,:SEQ_LEN_TRAIN]
                    label = label[:,:SEQ_LEN_TRAIN]
                else:
                    oup = out[:, 0].to(device)
                    label = label

                loss_main = criterion(oup, label)

                loss_tim = class_criterion(tim_out, tim_cls.long())
                loss_typ = class_criterion(typ_out, typ_cls.long())

                loss = LOSS_MAIN * loss_main + LOSS_TIM * loss_tim + LOSS_TYP * loss_typ #

                loss.backward()
                optimizer.step()

                net.eval()
                out, tim_out, typ_out = net(ave, ave_q, con)

                _, out_tim = torch.max(torch.softmax(tim_out, 1), 1)
                out_tim = out_tim.cpu().numpy()
                cls_tim = tim_cls.long().cpu().numpy()
                que_score = round(metrics.accuracy_score(out_tim, cls_tim) * 100, 2)


                net.train()

                if it % 20 == 0:
                    c_lr = scheduler.get_last_lr()
                    loss_info = 'TOTAL:{:.6f},Main: {:.6f},Class: {:.6f}'.format(loss.item(), loss_main.item(), loss_tim.item())
                    info = '-- Iter:{},Loss:{},Class:{},lr:{}'.format(it, loss_info, que_score, c_lr)
                    print(info)
                    record.write(info + '\n')

                if it % 20 == 0:
                    rmse = VALRMSE(oup, label, ds_factory.ds, ds_factory.dataset.m_factor)
                    info_matrix = "[epoch %d][%d/%d] mae: %.4f rmse: %.4f" % (
                        epoch, i + 1, len(train_loader), loss_main.item(), rmse.item())
                    record.write(info_matrix + '\n')
                    print(info_matrix)

                it += 1

            t = timeSince(start)
            loss_info = 'D:{:.6f}'.format(loss.item())
            info = 'EPOCH:{}/{},Loss {}, Time {}'.format(epoch, EPOCH_E, loss_info, t)
            print(info)
            record.write(info + '\n')
            scheduler.step()

            if (epoch + 1) % 1 == 0:

                dirs = './model/Imp_{}'.format(RECORD_ID)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                model_path = os.path.join(dirs, f'pre_model_{epoch + 1}.pth')

                if TORCH_VERSION == '1.6.0' or TORCH_VERSION == '1.7.0':
                    torch.save(net.cpu().module.state_dict(), model_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(net.cpu().module.state_dict(), model_path)

                net = net.to(device)

        record.close()

    if IS_VAL:
        ### TEST DATASET ###
        test_ds = ds_factory.get_test_dataset()

        if IS_BEST:
            EVAL_BATCH = 1
            EPOCH_E = EVAL_START_EPOCH + 1

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        #### MODEL ####


        print('EVALUATION START')
        print('-' * 30)

        record = open("record/{}/log_eval.txt".format(RECORD_ID), "w")  ###xie

        if 1:
            rmse_list = []  ###xie
            mae_list = []  ###xie
            for epoch in range(EVAL_START_EPOCH, EPOCH_E):

                net = Model(
                    mcof, Length=LENGTH, Width=W, Height=H, Input_dim=C,
                    Patch_list=PATCH_LIST, Dropout=DROPOUT, Att_num=ATT_NUM,
                    Cross_att_num=CROSS_ATT_NUM, Using_skip=IS_USING_SKIP,
                    Encoding_dim=MODEL_DIM, Embedding_dim=MODEL_DIM,
                    Is_mask=IS_MASK_ATT, Is_reduce=IS_REDUCE,
                    Debugging=0,Merge=MERGE,
                )

                model_path = './model/Imp_{}/pre_model_{}.pth'.format(RECORD_ID, epoch + 1)
                print(model_path)

                net.load_state_dict(torch.load(model_path))

                device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                net = net.to(device)
                net = nn.DataParallel(net)

                criterion = nn.MSELoss().to(device)

                net.eval()
                mse = 0.0
                mse_in = 0.0
                mse_out = 0.0
                mae = 0.0
                target = []
                pred = []
                mmn = ds_factory.ds.mmn
                if IS_RECT:
                    m_factor = ds_factory.dataset.m_factor
                else:
                    m_factor = 1.0


                if IS_BEST:
                    test_rmse_list = []
                ts_Y_test = ds_factory.ds.ts_Y_test
                with torch.no_grad():
                    for i, data in enumerate(test_loader, 0):

                        # (B,6,2,32,32)
                        con, ave, que, label, tim_cls, typ_cls = data


                        ave = ave.to(device)
                        que = que.to(device)
                        con = con.to(device)

                        gen_out, tim_out, typ_out = net(ave, que, con)

                        #eval

                        if IS_SEQ:
                            tar = label[:,:SEQ_LEN_TEST]
                            oup = gen_out[:, :SEQ_LEN_TEST]
                        else:
                            tar = label  ##niu
                            oup = gen_out[:, 0]
                        tar = tar.to(device)

                        loss = criterion(oup, tar)  # 所有样本损失的平均值

                        if IS_BEST:
                            rmse_ = math.sqrt(loss) * (mmn.max - mmn.min) / 2. * m_factor
                            print('->','timestamp',i,ts_Y_test[i],'rmse',rmse_)
                            test_rmse_list.append(rmse_)

                        mse += con.shape[0] * loss.item()  # 所有样本损失的总和
                        mae += con.shape[0] * torch.mean(
                            torch.abs(oup - tar)).item()  # mean()不加维度时，返回所有值的平均

                        ##niu
                        mse_in += con.shape[0] * torch.mean(
                            (tar[:, 0] - oup[:, 0]) * (tar[:, 0] - oup[:, 0])).item()
                        mse_out += con.shape[0] * torch.mean(
                            (tar[:, 1] - oup[:, 1]) * (tar[:, 1] - oup[:, 1])).item()

                        _, out_cls = torch.max(torch.softmax(tim_out, 1), 1)
                        out_class = out_cls.cpu().numpy()
                        lab_class = tim_cls.long().cpu().numpy()

                        target.append(lab_class)
                        pred.append(out_class)

                    if IS_BEST:
                        np.save('test/seq{}_our_short_test_bikenyc.npy'.format(SEQ_LEN_TEST),np.stack(test_rmse_list))
                        f = open("test/short_bikenyc_timestamp.pkl",'wb')
                        pickle.dump(ts_Y_test[:24],f)
                        f.close()

                lab_c = np.concatenate(target)
                oup_c = np.concatenate(pred)
                acc = metrics.accuracy_score(oup_c, lab_c) * 100

                ## Validation
                cnt = ds_factory.ds.X_con_tes.shape[0]

                mae /= cnt
                mae = mae * (mmn.max - mmn.min) / 2. * m_factor

                mse /= cnt
                rmse = math.sqrt(mse) * (mmn.max - mmn.min) / 2. * m_factor

                rmse_list.append(rmse)  ##xie
                mae_list.append(mae)  ##xie

                mse_in /= cnt
                rmse_in = math.sqrt(mse_in) * (mmn.max - mmn.min) / 2. * m_factor
                mse_out /= cnt
                rmse_out = math.sqrt(mse_out) * (mmn.max - mmn.min) / 2. * m_factor

                info = "MAE:%.4f RMSE:%.4f in RMSE: %.5f out RMSE: %.4f ACC: %.2f" % (mae,rmse,rmse_in, rmse_out,acc)  ###xie
                print(info)  ###xie
                record.write(info + '\n')  ###xie

                min_idx = rmse_list.index(min(rmse_list))  ###xie
                rmse_min = round(rmse_list[min_idx], 2)  ###xie
                mae_min = round(mae_list[min_idx], 2)  ###xie
                info = '- Best:RMSE:{},MAE:{},epoch:{}'.format(rmse_min, mae_min, min_idx + 1)  ###xie
                #print('---------------------------------')  ###xie
                print(info)  ###xie
                #record.write('-----------------------' + '\n')  ###xie
                record.write(info + '\n')  ###xie

            record.close()  ###xie


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--mode', type=str, default='train', help='The processing phase of the model')
    parser.add_argument('--record', type=str, help='Recode ID')
    parser.add_argument('--presume_record', type=str, help='Presume Recode ID')
    parser.add_argument('--keep_train', type=int, default=0, help='Model keep training')
    parser.add_argument('--epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--presume_epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--inp_type', type=str, default='external',
                        choices=['external', 'train', 'accumulate', 'accumulate_avg', 'holiday', 'windspeed', 'weather',
                                 'temperature'])
    parser.add_argument('--patch_method', type=str, default='STTN', choices=['EINOPS', 'UNFOLD', 'STTN'])
    parser.add_argument('--dataset_type', type=str, default='All', choices=['Sub', 'All'],
                        help='datasets type is sub_datasets or all_datasets')
    parser.add_argument('--best', type=int, default=0, help='best test')
    parser.add_argument('--is_rect', type=int, default=1, help='is_rect')
    mcof = parser.parse_args()

    run(mcof)
