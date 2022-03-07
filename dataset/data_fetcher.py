import numpy as np
import torch
import time

from dataset.utils import string2timestamp


class DataFetcher(object):
    """
    construct XC, XP, XT, Y

    current timestamp - offset = timestamp of interest
    data = fetchdata (timestamp of interest)
    """

    def __init__(self, data, raw_ts, avg_data, ext_cls, t=48):
        assert len(data) == len(raw_ts)

        super(DataFetcher, self).__init__()

        self.data = data
        self.average_data = avg_data
        self.external_cls = ext_cls
        self.raw_ts = raw_ts
        self.T = t

        # get time offset between adjacent slices
        assert (t == 24 or t == 48, 'T should be 24 / 48 or the code have to be edit')
        self.offset_frame = np.timedelta64(24 * 60 // self.T, 'm')
        self.ts = string2timestamp(self.raw_ts, self.offset_frame)  # convert to timestamp

        # print('catch ts:\n', self.ts[288:384])   2013-07-07 2013-07-09
        # create index
        self.idx_of_ts = dict()
        for idx, _ts in enumerate(self.ts):
            self.idx_of_ts[_ts] = idx

    def dat_of_matrix(self, ts_mat):
        dat_mat = [[self.data[self.idx_of_ts[ts]] for ts in ts_seq] for ts_seq in ts_mat]
        x_c = dat_mat[0]
        x_p = dat_mat[1]
        x_t = dat_mat[2]

        return x_c, x_p, x_t

    def is_matrix_valid(self, ts_mat):
        """
        validation for timestamp matrix
        """
        for ts_seq in ts_mat:
            for ts in ts_seq:
                if ts not in self.idx_of_ts.keys():
                    return False, ts
        return True, None

    def create_timestamp_matrix(self, cur_ts, offset_mat):
        """
        get all pd_ts sequence of interest according current pd_ts and all pd_ts offset sequence
        pd_ts: pandas timestamp
        pto: pandas timestamp offset
        ptos: pandas timestamp offset sequence
        all_ptos: all pandas timestamp offset sequence from Closeness, Period and Trend
        """
        # closeness sequence length is 4, take the first 4 interval
        # Period sequence length is 2, take the first two days
        timestamp_matrix = \
            [
                [
                    cur_ts - offset * self.offset_frame
                    for offset in offset_seq
                ]
                for offset_seq in offset_mat      # 双重嵌套list generator
            ]

        return timestamp_matrix

    def fetch_data(self, dconf):
        """
        construct the array of data while doing validation
        """
        lc = dconf.len_close  # 3
        lp = dconf.len_period  # 2
        lt = dconf.len_trend  # 0
        fp = dconf.pad_forward_period  # 0
        bp = dconf.pad_back_period  # 0
        ft = dconf.pad_forward_trend  # 0
        bt = dconf.pad_back_trend  # 0
        ip = dconf.interval_period  # 1
        it = dconf.interval_trend  # 7
        print('  DataFetcher:',
              'With Length: %d, %d, %d; ' % (lc, lp, lt),
              'with Padding: %d %d, %d %d; ' % (fp, bp, ft, bt),
              'with Interval: %d %d.' % (ip, it))
        '''
        Structure:
              |<---------------------- it --------------------->|
              |                     |<----------- ip ---------->|
       -+--+--+--+--+--+-    -+--+--+--+--+--+-    -+--+--+--+--+-
        |  |  |  |  |  | .... |  |  |  |  |  | .... |  |  |  |  | < Y data
       -+--+--+--+--+--+-    -+--+--+--+--+--+-    -+--+--+--+--+-
        |<bt >|  |< ft>|      |<bp >|  |< fp>|      |<- lc ->|
         ^  ^  ^  ^  ^         ^  ^  ^  ^  ^          ^  ^  ^
        Trend data x lt       Period data x lp     Closeness data

        '''
        # get all offset
        # aptos: the offset sequence to indicate the pd_ts of interest
        # interval: interval of day between slices
        # current timestamp - offset = timestamp of interest
        # flatten a list using 2 'for'
        r_c = range(1, lc + 1)
        rl_p = [range(ip * self.T * i - fp, ip * self.T * i + bp + 1) for i in range(1, lp + 1)]
        rl_t = [range(it * self.T * i - ft, it * self.T * i + bt + 1) for i in range(1, lt + 1)]

        # [[1, 2, 3, 4], [48, 96], []]
        offset_mat = \
            [
                [e for e in r_c],                     # closeness
                [e for r_p in rl_p for e in r_p],     # period
                [e for r_t in rl_t for e in r_t]      # trend
            ]

        # fetching loop
        x, y, x_ave, y_cls, ts_x, ts_y, ts_interval = [], [], [], [], [], [], []
        ts_dumped = []
        # 96
        largest_interval = max([k[-1] if len(k) is not 0 else 0 for k in offset_mat])  # init using the largest interval
        for cur_ts in self.ts[largest_interval:]:
            # cur_ts:2013-07-03T00:00
            ts_mat = self.create_timestamp_matrix(cur_ts, offset_mat)  # list nest list

            # timestamp validation
            flag, pos = self.is_matrix_valid(ts_mat)
            if flag is False:
                ts_dumped.append((cur_ts, pos))
                continue

            x_c, x_p, x_t = self.dat_of_matrix(ts_mat)
            ave_dat = self.average_data[self.idx_of_ts[cur_ts]]
            ext_cls = self.external_cls[self.idx_of_ts[cur_ts]]
            # 4, 2, 0
            # concat as channel
            x_exist = [x_ for x_ in [x_c, x_p, x_t] if len(x_) > 0]
            x.append(np.vstack([np.stack(x_, axis=0) for x_ in x_exist]))
            y.append(self.data[self.idx_of_ts[cur_ts]])
            x_ave.append(ave_dat)
            y_cls.append(ext_cls)
            # ((12, 32, 32), (2, 32, 32))
            # ts_y: 2013-07-07T23:30
            ts_x.append(ts_mat[0] + ts_mat[1] + ts_mat[2])
            # ts_y中保存了具有完整所需信息的时间间隔
            ts_y.append(cur_ts)

        # concat to tensor
        x = np.asarray(x)
        y = np.asarray(y)
        x_ave = np.asarray(x_ave)
        y_cls = np.asarray(y_cls)

        X_inp_q = x_ave[:, :4]
        x_ave_q = np.concatenate([X_inp_q, x[:, -2:]], axis=1)

        print("    Dumped ", len(ts_dumped), " data.")
        return x, x_ave, x_ave_q, y, y_cls, ts_x, ts_y


if __name__ == '__main__':
    pass
