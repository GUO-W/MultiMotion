'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
'''

import json
import numpy as np
# from IPython import embed

def write_results_avg(exp_name='', table='ca', test_epo=24):
    results = json.load(open('./results.json', 'r'))
    if table == 'ca' or table == 'ua':
        for m in ['mpjpe_joi', 'mpjpe_ali']:
            res_print = exp_name+'_'+m+':\n'
            if table == 'ca':
                split_list= ['0', '1','2','3','4','5','6','AVG']
                t_list = [4,9,14,24]
            elif table == 'ua':
                split_list= ['6','7','8','0','1','2','3','4','5','AVG']
                t_list = [9,14,19]
            for s in split_list:
                for t in t_list:
                    key_exp = exp_name +'_testepo'+str(test_epo)
                    res_act = np.array(results[key_exp][s][m][t]).mean()
                    res_print = res_print + ' & ' + str(int(res_act))
            print(res_print)

    elif table == 'sa':
        for m in ['mpjpe_joi', 'mpjpe_ali']:
            res_print = exp_name+'_'+m+':\n'
            t_list = [4,9,14,24]
            sum_t = [0]*len(t_list)
            for s in ['0','1','2','3','4','5','6']:
                for i in range(len(t_list)):
                    t = t_list[i]
                    key_exp = exp_name.replace('NAME', s) +'_testepo'+str(test_epo)
                    res_act = np.array(results[key_exp]['AVG'][m][t]).mean()
                    res_print = res_print +' & ' + str(int(res_act))
            print(res_print)


if __name__ == '__main__':

    # TODO: modify pro and exp_name to generate different table. (all the tests needs to be runned first)
    # pro: 'ca': common-action-split; 'sa': single-action-split; 'ua': unseen-action-split
    # exp_name: for 'sa', use name '..._proNAME_...' instead of the real one '_pro0_'/'_pro1_'/...
    pro = 'sa'
    #exp_name = 'main_pi_3d_crossAtt_propro3_in50_kz10_lr0.005'
    exp_name = 'main_pi_3d_crossAtt_proNAME_in50_kz10_lr0.005'

    write_results_avg(exp_name, pro, test_epo=25)


