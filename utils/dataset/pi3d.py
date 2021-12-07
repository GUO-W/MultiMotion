'''
Software MultiMotion
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
MIT license.
'''
# pi3d.py

import torch
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils, vis_2p
from IPython import embed

class Datasets(Dataset):

    def __init__(self, opt, is_train=True):

        self.path_to_data = "./datasets/pi"
        self.is_train = is_train
        if is_train:#train
            self.in_n = opt.input_n
            self.out_n = opt.kernel_size
            self.split = 0
        else: #test
            self.in_n = 50
            self.out_n = opt.output_n
            self.split = 1
        self.skip_rate = 1
        self.p3d = {}
        self.data_idx = []

        if opt.protocol == 'pro3': # unseen action split
            if is_train: #train on acro2
                acts = ["2/a-frame","2/around-the-back","2/coochie","2/frog-classic","2/noser","2/toss-out", "2/cart-wheel",\
                        "1/a-frame","1/around-the-back","1/coochie","1/frog-classic","1/noser","1/toss-out", "1/cartwheel"]
                subfix = [[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[1,2,3,4,5],[2,3,4,5,6],\
                        [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,4,5,6],[1,2,3,4,6],[1,2,3,4,5],[3,4,5,6,7]]

            else: #test on acro1
                acts = ["2/crunch-toast", "2/frog-kick", "2/ninja-kick", \
                        "1/back-flip", "1/big-ben", "1/chandelle", "1/check-the-change", "1/frog-turn", "1/twisted-toss"]
                subfix = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                        [1,3,4,5,6],[1,2,3,4,5],[3,4,5,6,7],[1,2,4,5,8],[1,2,3,4,5],[1,2,3,4,5]]

                if opt.test_split is not None: #test per action for unseen action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]

        else: # common action split and single action split
            if is_train: #train on acro2
                acts = ["2/a-frame","2/around-the-back","2/coochie","2/frog-classic","2/noser","2/toss-out", "2/cartwheel"]
                subfix = [[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[3,4,5,6,7],[1,2,3,4,5],[1,2,3,4,5],[2,3,4,5,6]]

                if opt.protocol in ["0","1","2","3","4","5","6"]: # train per action for single action split
                    acts = [acts[int(opt.protocol)]]
                    subfix = [subfix[int(opt.protocol)]]

            else: #test on acro1
                acts = ["1/a-frame","1/around-the-back","1/coochie","1/frog-classic","1/noser","1/toss-out", "1/cartwheel"]
                subfix = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,4,5,6],[1,2,3,4,6],[1,2,3,4,5],[3,4,5,6,7]]

                if opt.test_split is not None: #test per action for common action split
                    acts, subfix = [acts[opt.test_split]], [subfix[opt.test_split]]
                if opt.protocol in ["0","1","2","3","4","5","6"]: #test per action for single action split
                    acts, subfix = [acts[int(opt.protocol)]], [subfix[int(opt.protocol)]]

        key = 0
        for action_idx in np.arange(len(acts)):
            subj_action = acts[action_idx]
            subj, action = subj_action.split('/')
            for subact_i in np.arange(len(subfix[action_idx])):
                subact = subfix[action_idx][subact_i]
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                filename = '{0}/acro{1}/{2}{3}/mocap_cleaned.tsv'.format(self.path_to_data, subj, action, subact)
                the_sequence = data_utils.readCSVasFloat(filename,with_key=True)
                num_frames = the_sequence.shape[0]
                the_sequence = data_utils.normExPI_2p_by_frame(the_sequence)
                the_sequence = torch.from_numpy(the_sequence).float().cuda()

                if self.is_train: #train
                    seq_len = self.in_n + self.out_n
                    valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)
                else: #test
                    seq_len = self.in_n + 30
                    valid_frames = data_utils.find_indices_64(num_frames, seq_len)

                p3d = the_sequence
                self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

        self.dimension_use = np.arange(18*2*3)
        self.in_features = len(self.dimension_use)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        data = self.p3d[key][fs][:,self.dimension_use]
        return data

