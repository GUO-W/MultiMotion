'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
'''
#opt.py

import os
import argparse
from pprint import pprint
from utils import util
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--exp', type=str, default='', help='experiment name')
        self.parser.add_argument('--ckpt', type=str, default=None , help='path of checkpoint')
        self.parser.add_argument('--tensorboard', type=str, default='tensorboard/', help='path to save tensorboard log')

        self.parser.add_argument('--model', type=str, default='crossAtt', help='model file used, for customze')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--dct_n', type=int, default=20, help='future frame number')

        self.parser.add_argument('--epoch', type=int, default=25)
        self.parser.add_argument('--batch_size', type=int, default=32)

        self.parser.add_argument('--lr_now', type=float, default=0.005)
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.98)
        self.parser.add_argument('--max_norm', type=float, default=10000)

        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true', help='test')
        #self.parser.add_argument('--val', dest='val', action='store_true', help='val')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',help='whether to load existing model')

        self.parser.add_argument('--input_n', type=int, default=50, \
                help='input frame number')
        self.parser.add_argument('--output_n', type=int, default=30, \
                help='len of prediction, itr/test = output_n/kernel_size_n, use in test')
        self.parser.add_argument('--kernel_size', type=int, default=10, \
                help='len of current observation, used in model.py')
        self.parser.add_argument('--sup_n', type=int, default=10, \
                help='len of supervised future when training, direct output len of one iteration in testing.')

        ##test
        self.parser.add_argument('--vis', dest ='vis',action='store_true',help='vis result')
        self.parser.add_argument('--save_results', dest ='save_results',action='store_true',help='save res in results.json')
        self.parser.add_argument('--test_epo', type=int, default=None, help='test epo')
        self.parser.add_argument('--test_split', type=int, default=None, help='for test protocol1')
        self.parser.add_argument('--test_batch_size', type=int, default=16)

        # different train/test split
        self.parser.add_argument('--protocol', type=str, default='pro1', help='pro1: common action split;\
                                                                                0-6: single action split;\
                                                                               pro3: unseen action split')


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        script_name = os.path.basename(sys.argv[0])[:-3]
        log_name = '{}_{}_pro{}_in{}_kz{}_lr{}'.format(script_name, self.opt.model, \
                self.opt.protocol, self.opt.input_n, self.opt.kernel_size, self.opt.lr_now)
        #for name in [self.opt.abla, self.opt.note, self.opt.aug]:
        #    if name is not None:
        #        log_name = log_name + '_{}'.format(name)

        if self.opt.is_eval:
            log_name = log_name.replace("_eval_","_") # load model when test
        self.opt.exp = log_name
        # do some pre-check
        if self.opt.ckpt is None:
            self.opt.ckpt = "./checkpoint"
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                util.save_options(self.opt)
            self.opt.ckpt = ckpt

        if not self.opt.is_eval:
            tensorboard = os.path.join(self.opt.tensorboard, self.opt.exp)
            if not os.path.isdir(tensorboard):
                os.makedirs(tensorboard)
                util.save_options(self.opt)
            self.opt.tensorboard = tensorboard

            util.save_options(self.opt)

        return self.opt
