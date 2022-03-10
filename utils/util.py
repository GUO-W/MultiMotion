'''
adapted from : https://github.com/wei-mao-2019/HisRepItself/blob/master/model/GCN.py
under MIT license.
'''
# util.py

from __future__ import absolute_import
import json
import os
import torch
import pandas as pd
import numpy as np

def lr_decay_step(optimizer, epo, lr, gamma):
    if epo % 3 == 0:
        lr = lr * gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def lr_decay_mine(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def save_csv_log(opt, head, value, is_create=False, file_name='train_log'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = opt.ckpt + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(state, epo, opt=None):
    file_path = os.path.join(opt.ckpt, 'ckpt_last.pth.tar')
    torch.save(state, file_path)
    # if epo ==24: # % 4 == 0 or epo>22 or epo<5:
    if epo % 5 == 0:
        file_path = os.path.join(opt.ckpt, 'ckpt_epo'+str(epo)+'.pth.tar')
        torch.save(state, file_path)


def save_options(opt):
    with open(opt.ckpt + '/option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))

