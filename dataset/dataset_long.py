import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
import pdb
import time

from dataset.minmax_normalization import MinMaxNormalization
from dataset.data_fetcher_long import DataFetcher


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    print('*' * 10 + 'DEBUG' + '*' * 10)
    print(datapath)

    def __init__(self, dconf, Inp_type, Data_type, Length, Is_seq, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.dataset = dconf.name
        self.len_close = dconf.len_close
        self.len_period = dconf.len_period
        self.len_trend = dconf.len_trend
        self.datapath = datapath
        self.inp_type = Inp_type
        self.data_type = Data_type
        self.length = Length
        self.is_seq = Is_seq

        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ/dataset'
            if self.data_type == 'Sub':
                self.dataname = [
                    'BJ16_M32x32_T30_InOut.h5'
                ]
            else:
                self.dataname = [
                    'BJ13_M32x32_T30_InOut.h5',
                    'BJ14_M32x32_T30_InOut.h5',
                    'BJ15_M32x32_T30_InOut.h5',
                    'BJ16_M32x32_T30_InOut.h5'
                ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            if self.len_close == 4 and self.len_period == 2 and self.len_trend == 2:
                test_days = 24 if test_days == -1 else test_days
            else:
                test_days = 28 if test_days == -1 else test_days

            self.m_factor = 1.

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 16 * 8 / 81)

        elif self.dataset == 'TaxiNYC':
            self.datafolder = 'TaxiNYC'
            self.dataname = ['NYC2014.h5']
            self.nb_flow = 2
            self.dim_h = 15
            self.dim_w = 5
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 15 * 5 / 64)

        else:
            raise ValueError('Invalid dataset')

        self.len_test = test_days * self.T
        self.portion = dconf.portion

    def get_raw_data(self):
        """
         data:
         np.array(n_sample * n_flow * height * width)
         ts:
         np.array(n_sample * length of timestamp string)
        """
        raw_data_list = list()
        raw_ts_list = list()
        print("  Dataset: ", self.datafolder)

        for filename in self.dataname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, 'raw_data',filename), 'r')
            _raw_data = f['data'][()]
            _raw_ts = f['date'][()]
            f.close()

            raw_data_list.append(_raw_data)
            raw_ts_list.append(_raw_ts)
        # delete data over 2channels

        return raw_data_list, raw_ts_list

    @staticmethod
    def remove_incomplete_days(data, timestamps, t=48):
        print("before removing", len(data))
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + t - 1 < len(timestamps) and int(timestamps[i + t - 1][8:]) == t:
                days.append(timestamps[i][:8])
                i += t
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        print("after removing", len(data))
        return data, timestamps

    def trainset_of(self, vec):
        return vec[:math.floor((len(vec) - self.len_test + 10) * self.portion)]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion) + 10:]

    def split(self, x, y, x_ave, x_ave_q, y_cls, y_typ):
        x_tra = self.trainset_of(x)
        x_tes = self.testset_of(x)

        x_ave_tra = self.trainset_of(x_ave)
        x_ave_tes = self.testset_of(x_ave)

        x_ave_q_tra = self.trainset_of(x_ave_q)
        x_ave_q_tes = self.testset_of(x_ave_q)

        y_tra = self.trainset_of(y)
        y_tes = self.testset_of(y)

        y_tra_cls = self.trainset_of(y_cls)
        y_tes_cls = self.testset_of(y_cls)

        y_tra_typ = self.trainset_of(y_typ)
        y_tes_typ = self.testset_of(y_typ)

        return x_tra, x_ave_tra, x_ave_q_tra, y_tra, y_tra_cls, y_tra_typ, x_tes, x_ave_tes, x_ave_q_tes, y_tes, y_tes_cls, y_tes_typ

    def load_data(self):
        """
        return value:
            X_train & X_test: [XC, XP, XT, Xext]
            Y_train & Y_test: vector
        """
        # read file and place all of the raw data in np.array. 'ts' means timestamp
        # without removing incomplete days
        print('Preprocessing: Reading HDF5 file(s)')
        raw_data_list, ts_list = self.get_raw_data()

        # filter dataset
        data_list, ts_new_list = [], []
        for idx in range(len(ts_list)):
            raw_data = raw_data_list[idx]
            ts = ts_list[idx]

            if self.dconf.rm_incomplete_flag:
                raw_data, ts = self.remove_incomplete_days(raw_data, ts, self.T)

            data_list.append(raw_data)  # 列表套列表套数组，最外层长度为4
            ts_new_list.append(ts)

        # 1、归一化数据如何求方差和均值，在整个数据集上还是训练集上
        # 2、求平均是在整个数据集上还是训练集上
        # (21360, 6, 2, 32, 32)   21360/48 = 445

        print(f'=============={self.inp_type} 输入加载成功！=============')
        #inp_path = f'./data/BikeNYC/{self.data_type}set/AVG{self.length}/{self.inp_type}_inp_average.npy'
        inp_path = f'./data/BikeNYC/AVG6_4/expectation_inp.npy'
        all_average_data = np.load(inp_path, allow_pickle=True)
        new_average_data_list = list(all_average_data)

        #ext_cls_path = f'./data/BikeNYC/{self.data_type}set/AVG{self.length}/{self.inp_type}_cls.npy'
        ext_cls_path = f'./data/BikeNYC/AVG6_4/expectation_cls.npy'
        all_ext_cls = np.load(ext_cls_path, allow_pickle=True)
        new_all_ext_cls = list(all_ext_cls)

        print('Preprocessing: Min max normalizing')
        raw_data = np.concatenate(data_list)
        mmn = MinMaxNormalization()
        # # (21360, 2, 32, 32), len(ts_new_list)=4
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]
        print('Context data min max normalizing processing finished!')

        x_con_list, y_list, x_ave_list, x_ave_q_list, y_typ_list, ts_x_list, ts_y_list = [], [], [], [], [], [], []
        for idx in range(len(ts_new_list)):
            x_con, x_ave, x_ave_q, y, y_typ, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], new_average_data_list[idx], new_all_ext_cls[idx],
                            self.len_test, self.T).fetch_data(self.dconf)
            x_con_list.append(x_con)
            y_list.append(y)

            x_ave_list.append(x_ave)
            y_typ_list.append(y_typ)
            x_ave_q_list.append(x_ave_q)

            ts_x_list.append(ts_x)  # list nest list nest list nest numpy.datetime64 class
            ts_y_list.append(ts_y)  # list nest list nest numpy.datetime64 class
        x_con = np.concatenate(x_con_list)
        y = np.concatenate(y_list)
        x_ave = np.concatenate(x_ave_list)
        x_ave_q = np.concatenate(x_ave_q_list)
        y_typ = np.concatenate(y_typ_list)
        ts_y = np.concatenate(ts_y_list)

        print(ts_y[9:][-self.len_test+9:])

        Y_Class = []
        for i in range(len(ts_new_list)):
            Y_Class.append(np.array(range(10, 24)))
            for j in range(len(ts_y_list[i])//24-1):
                Y_Class.append(np.array(range(0, 24)))
        y_cls = np.concatenate(Y_Class, axis=0).reshape(-1, 1)

        y_typ = y_typ.reshape(-1, 1)

        # (16464, 12, 32, 32) (16464, 2, 32, 32) (16464, 6) (16464,)
        x_con_tra, x_ave_tra, x_ave_q_tra, y_tra, y_cls_tra, y_typ_tra, x_con_tes, x_ave_tes, x_ave_q_tes, y_tes, y_cls_tes, y_typ_tes = self.split(
            x_con, y, x_ave, x_ave_q, y_cls, y_typ)

        x_con_tes = x_con_tes[:-14]
        x_ave_tes = x_ave_tes[:-14]
        x_ave_q_tes = x_ave_q_tes[:-14]
        y_tes = y_tes[:-14]
        y_cls_tes = y_cls_tes[:-14]
        y_typ_tes = y_typ_tes[:-14]

        # 是否使用多个序列长度求loss
        if self.is_seq:
            x_con_tra = x_con_tra[:-self.length + 1]
            x_con_tes = x_con_tes[:-self.length + 1]
            x_ave_tra = x_ave_tra[:-self.length + 1]
            x_ave_tes = x_ave_tes[:-self.length + 1]
            x_ave_q_tra = x_ave_q_tra[:-self.length + 1]
            x_ave_q_tes = x_ave_q_tes[:-self.length + 1]
            y_cls_tra = y_cls_tra[:-self.length + 1]
            y_cls_tes = y_cls_tes[:-self.length + 1]
            y_typ_tra = y_typ_tra[:-self.length + 1]
            y_typ_tes = y_typ_tes[:-self.length + 1]

            Y_seq_tra = []
            for i, _ in enumerate(y_tra[:-self.length + 1]):
                Y_seq_tra.append(y_tra[i:i + self.length])
            y_seq_tra = np.stack(Y_seq_tra)

            Y_seq_tes = []
            for i, _ in enumerate(y_tes[:-self.length + 1]):
                Y_seq_tes.append(y_tes[i:i + self.length])
            y_seq_tes = np.stack(Y_seq_tes)

        class TempClass:
            def __init__(self_2):
                self_2.X_con_tra = x_con_tra
                self_2.X_ave_tra = x_ave_tra
                self_2.X_ave_q_tra = x_ave_q_tra
                if self.is_seq:
                    self_2.Y_tra = y_seq_tra
                else:
                    self_2.Y_tra = y_tra
                self_2.Y_cls_tra = y_cls_tra
                self_2.Y_typ_tra = y_typ_tra

                self_2.X_con_tes = x_con_tes
                self_2.X_ave_tes = x_ave_tes
                self_2.X_ave_q_tes = x_ave_q_tes
                if self.is_seq:
                    self_2.Y_tes = y_seq_tes
                else:
                    self_2.Y_tes = y_tes
                self_2.Y_cls_tes = y_cls_tes
                self_2.Y_typ_tes = y_typ_tes

                self_2.img_mean = np.mean(train_dat, axis=0)
                self_2.img_std = np.std(train_dat, axis=0)
                self_2.mmn = mmn
                self_2.ts_Y_train = self.trainset_of(ts_y)
                self_2.ts_Y_test = self.testset_of(ts_y)

            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_con_tra.shape, self_2.X_ave_tra.shape, self_2.X_ave_q_tra.shape,
                    self_2.X_con_tes.shape, self_2.X_ave_tes.shape, self_2.X_ave_q_tes.shape,
                    "Y inputs shape: ", self_2.Y_tra.shape, self_2.Y_cls_tra.shape, self_2.Y_typ_tra.shape,
                    self_2.Y_tes.shape, self_2.Y_cls_tes.shape, self_2.Y_typ_tes.shape
                )
                print("Run: min~max: ", self_2.mmn.min, '~', self_2.mmn.max)

        return TempClass()


class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train'):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            X_con = torch.from_numpy(self.ds.X_con_tra[index])
            X_ave = torch.from_numpy(self.ds.X_ave_tra[index])
            X_ave_q = torch.from_numpy(self.ds.X_ave_q_tra[index])
            Y = torch.from_numpy(self.ds.Y_tra[index])
            Y_tim = torch.Tensor(self.ds.Y_cls_tra[index])
            Y_typ = torch.Tensor(self.ds.Y_typ_tra[index])
        else:
            X_con = torch.from_numpy(self.ds.X_con_tes[index])
            X_ave = torch.from_numpy(self.ds.X_ave_tes[index])
            X_ave_q = torch.from_numpy(self.ds.X_ave_q_tes[index])
            Y = torch.from_numpy(self.ds.Y_tes[index])
            Y_tim = torch.Tensor(self.ds.Y_cls_tes[index])
            Y_typ = torch.Tensor(self.ds.Y_typ_tes[index])

        return X_con.float(), X_ave.float(), X_ave_q.float(), Y.float(), Y_tim.float(), Y_typ.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_con_tra.shape[0]
        else:
            return self.ds.X_con_tes.shape[0]


class DatasetFactory(object):
    def __init__(self, dconf, Inp_type, Data_type, Length, Is_seq):
        self.dataset = Dataset(dconf, Inp_type, Data_type, Length, Is_seq)
        self.ds = self.dataset.load_data()
        print('Show a list of dataset!')
        print(self.ds.show())

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')

    def get_test_dataset(self):
        return TorchDataset(self.ds, 'test')


if __name__ == '__main__':
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


    df = DatasetFactory(DataConfiguration(0, 1, 1), Inp_type='train', Data_type='All', Length=6, Is_seq=0)
    ds = df.get_train_dataset()
    X, X_ave, X_ave_q, Y, Y_cls, Y_typ = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_ave.size())
    print(X_ave_q.size())
    print(Y.size())
    print(Y_cls)
    print(Y_typ)

    # ds = df.get_train_dataset()
    # X, X_ave, X_ext, Y, Y_ext = next(iter(ds))
    # print('test:')
    # print(X.size())
    # print(X_ave.size())
    # print(X_ext.size())
    # print(Y.size())
    # print(Y_ext.size())
