'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
'''
# data_utils.py
# func utils for data

import numpy as np
import torch
# from IPython import embed

###########################################
## func for reading data

def readCSVasFloat(filename, with_key=True):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    if with_key: # skip first line
        lines = lines[1:]
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

###########################################
## func utils for norm/unnorm

def normExPI_xoz(img, P0,P1,P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: olane xoz

    X0 = P0
    X1 = (P1-P0) / np.linalg.norm((P1-P0)) + P0 #x
    X2 = (P2-P0) / np.linalg.norm((P2-P0)) + P0
    X3 = np.cross(X2-P0, X1-P0) + P0 #y
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1-P0, X3-P0) + P0 #z

    X = np.concatenate((np.array([X0,X1,X2,X3]).transpose(), np.array([[1, 1, 1,1]])), axis = 0)
    Q = np.array([[0,0,0],[1,0,0],[0,0,1], [0,1,0]]).transpose()
    M  = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp,np.array([1])),axis=0)
        img_norm[i] =  M.dot(tmp)
    return img_norm

def normExPI_2p_by_frame(seq):
    nb, dim = seq.shape # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1,3)) #36
        P0 = (img[10] + img[11])/2
        P1 = img[11]
        P2 = img[3]
        img_norm = normExPI_xoz(img, P0,P1,P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm

def unnorm_abs2Indep(seq):
    # in:  torch.size(bz, nb_frames, 36, 3)
    # out: torch.size(bz, nb_frames, 36, 3)
    seq = seq.detach().cpu().numpy()
    bz, frame, nb, dim = seq.shape
    seq_norm = seq
    for j in range(bz):
        for i in range(frame):
            img = seq[j][i]

            P0_m = (img[10] + img[11])/2
            P1_m = img[11]
            P2_m = img[3]
            if nb == 36:
                img_norm_m = normExPI_xoz(img[:int(nb/2)], P0_m,P1_m,P2_m)
                P0_f = (img[18+10] + img[18+11])/2
                P1_f = img[18+11]
                P2_f = img[18+3]
                img_norm_f = normExPI_xoz(img[int(nb/2):], P0_f,P1_f,P2_f)
                img_norm = np.concatenate((img_norm_m, img_norm_f))
            elif nb == 18:
                img_norm = normExPI_xoz(img, P0_m,P1_m,P2_m)
            seq_norm[j][i] = img_norm.reshape((nb,dim))
    seq = torch.from_numpy(seq_norm).cuda()
    return seq


###########################################
## func utils for finding test samples

def find_indices_64(num_frames, seq_len):
    # not random choose. as the sequence is short and we want the test set to represent the seq better
    seed = 1234567890
    np.random.seed(seed)

    T = num_frames - seq_len + 1
    n = int(T / 64)
    list0 = np.arange(0,T)
    list1 = np.arange(0,T,(n+1))
    t =  64 - len(list1)
    if t == 0:
        listf = list1
    else:
        list2 = np.setdiff1d(list0, list1)
        list2 = list2[:t]
        listf = np.concatenate((list1, list2))
    return listf


