'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPLlicense.
'''
# main_pi_3d.py
# train & test on ExPI

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# from IPython import embed
from tqdm import tqdm

from utils.opt import Options
from utils import util, data_utils, vis_2p
from utils.rigid_align import rigid_align_torch
from utils.dataset import pi3d as datasets
from model import AttModel_crossAtt_unshare as AttModel

def main(opt):
    torch.manual_seed(1234567890)

    ## dataset
    print('>>> DATA loading >>>')
    if not opt.is_eval: #train
        dataset = datasets.Datasets(opt, is_train=True)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else: #test
        dataset = datasets.Datasets(opt, is_train=False)
        print('>>> Test dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    in_features = dataset.in_features
    nb_kpts = int(in_features/3)

    ## model
    print('>>> MODEL >>>')
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=opt.kernel_size, d_model=opt.d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n, input_n=opt.input_n)
    net_pred.cuda()
    lr_now = opt.lr_now
    start_epoch = 1
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    print(">>> is_load:{}, is_eval:{}, ckpt:{}".format(opt.is_load, opt.is_eval,  opt.ckpt))
    if opt.is_load or opt.is_eval:
        if '.pth.tar' in opt.ckpt:
            model_path_len = opt.ckpt
        elif opt.test_epo is not None:
            model_path_len = '{}/ckpt_epo{}.pth.tar'.format(opt.ckpt, opt.test_epo)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | err: {} | lr: {})".format(ckpt['epoch'], ckpt['err'], lr_now))

    ## train or test
    if not opt.is_eval: #train
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
        if opt.is_load:
            optimizer.load_state_dict(ckpt['optimizer'])
        util.save_ckpt({'epoch': 0, 'lr': lr_now, 'err': 0, 'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()}, 0, opt=opt)
        writer = SummaryWriter(opt.tensorboard)

        for epo in tqdm(range(start_epoch, opt.epoch + 1)):
            ret_train = run_model(nb_kpts, net_pred, optimizer, data_loader=data_loader, opt=opt,is_train=1,epo=epo)
            writer.add_scalar('scalar/train', ret_train['loss_train'], epo)

            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            # lr_log = lr_now*pow(10,-(epo-1))
            #writer.add_scalar('scalar/learning_rate', lr_log, epo)

            print('epo{}, train error: {:.3f}, lr: {:.6f}'.format(epo, ret_train['loss_train'], lr_now))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            util.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            util.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_train['loss_train'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            epo, opt=opt)
        writer.close()

    else: #test
        run_model(nb_kpts, net_pred, data_loader=data_loader, opt=opt, is_train=0)

def run_model(nb_kpts, net_pred, optimizer=None, data_loader=None, opt=None, is_train=0, epo=0):
    # is_train: 0 test, 1 train, 2 val

    raw_in_n = 50 #use this fix number in dataloader to ensure the same test data when changing opt.input_n
    in_n = opt.input_n #real input len  ( <=50 )
    in_n_s = raw_in_n - in_n # input start nb.

    sup_n = opt.sup_n
    kz = opt.kernel_size
    out_n =  opt.output_n #prediction len (for test)

    n = 0
    if is_train == 1: #train
        net_pred.train()
        loss_train = 0
        for i, (data_) in enumerate(data_loader): # in_n + kz

            batch_size, seq_len, _ = data_.shape
            if batch_size == 1:
                continue #when only one sample in this batch
            n += batch_size

            data_ = data_.float().cuda()
            data_in = data_[:, :-kz] #bz,in_n, 108
            ### run
            data_out = net_pred(data_in)[:,:,0] #bz, 2kz, 108
            data_gt = data_[:, -kz-kz:] #len of current observation+direct output len=20

            ## loss
            loss_l = torch.mean(torch.norm((data_out- data_gt)[:,:,:54], dim=2))
            loss_f = torch.mean(torch.norm((data_out- data_gt)[:,:,54:], dim=2))
            loss = loss_f + loss_l * pow(10,-(epo-1))

            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.cpu().data.numpy() * batch_size
            optimizer.step()
        res_dic = {"loss_train" : loss_train / n }
        return res_dic

    else: #test
        net_pred.eval()
        mpjpe_joi, mpjpe_ali = np.zeros([out_n]),np.zeros([out_n])
        for i, (data_) in enumerate(data_loader): # raw_in_n + out_n
            batch_size, seq_len, _ = data_.shape #raw_in_n+30
            n += batch_size

            data_ = data_.reshape([-1, seq_len, nb_kpts*3]).float().cuda() #raw_in_n + 30
            data_in = data_[:, in_n_s:raw_in_n] #in_n
            gt = data_[:, raw_in_n:].reshape([-1, out_n, nb_kpts,3]) #out_n

            itr_test = int(out_n/kz) + 1
            pred = []
            for itera in range(itr_test):
                data_out = net_pred(data_in) #, input_n=in_n, output_n=2*kz, itera=1)#bz,10+10,1,108
                data_out = data_out[:, kz:,0] #batch_size, sup_n, nb_joints*joint_size
                pred.append(data_out)
                data_in = torch.cat((data_in[:,kz:], data_out),axis=1)
            pred = torch.cat(pred, axis=1)[:,:out_n].reshape([-1, out_n, nb_kpts, 3]) #batch_size, out_len, nb_joints, joint_size

            ### evaluate###
            ## JME
            tmp_joi = torch.sum(torch.mean(torch.norm(gt - pred, dim=3), dim=2), dim=0)
            mpjpe_joi += tmp_joi.cpu().data.numpy()
            if opt.vis:
                vis_2p.vis_pi_compare(data_.reshape([-1, seq_len, nb_kpts,3]).detach().cpu().numpy()[0],\
                        pred_all.detach().cpu().numpy()[0], './outputs/example.mp4')

            ## AME
            pred_ali = pred #torch.cat((pred[:,:,int(nb_kpts/2):], data_utils.unnorm_abs2Indep(pred[:,:,:int(nb_kpts/2)])), axis=2)
            gt_ali = gt #torch.cat((gt[:,:,int(nb_kpts/2):], data_utils.unnorm_abs2Indep(gt[:,:,:int(nb_kpts/2)])), axis=2)

            pred_ali_l = rigid_align_torch(pred_ali[:,:,:int(nb_kpts/2)], gt_ali[:,:,:int(nb_kpts/2)])
            pred_ali_f = rigid_align_torch(pred_ali[:,:,int(nb_kpts/2):], gt_ali[:,:,int(nb_kpts/2):])
            pred_ali = torch.cat((pred_ali_l,pred_ali_f), axis=2)

            tmp_ali = torch.sum(torch.mean(torch.norm(gt_ali - pred_ali, dim=3), dim=2), dim=0)
            mpjpe_ali += tmp_ali.cpu().data.numpy()

        mpjpe_joi, mpjpe_ali = mpjpe_joi/n, mpjpe_ali/n  # n = testing dataset length

        out_print_frame = [4,9,14,19,24]
        res_dic = {"mpjpe_joi": mpjpe_joi[out_print_frame],
                   "mpjpe_ali": mpjpe_ali[out_print_frame]}
        print('Error at each output frame:\n Frame number:{}\n Error:{}'.format(out_print_frame,res_dic))

        if opt.save_results:
            # save result of all experiments together to easier generate the result tables.
            import json
            results = json.load(open('./outputs/results.json', 'r'))
            key_exp = opt.exp+ '_testepo'+str(opt.test_epo)
            print('save name exp:', opt.exp)
            if key_exp not in results:
                results[key_exp]={}
            ts = str(opt.test_split) if opt.test_split is not None else 'AVG'
            if ts not in results[key_exp]:
                results[key_exp][ts]={}
            results[key_exp][ts]={"mpjpe_joi": mpjpe_joi.tolist(),
                                  "mpjpe_ali": mpjpe_ali.tolist()}
            with open('outputs/results.json', 'w') as w:
                json.dump(results, w)

        return res_dic


if __name__ == '__main__':
    option = Options().parse()
    main(option)


