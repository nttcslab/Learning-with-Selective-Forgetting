from argparse import ArgumentParser
import numpy as np
import torch
import datetime
import os
import subprocess
# import read_config as read_config
from PIL import Image
import glob
import re
import pickle

# import src.utils as utils
def out_res(pdir,dirs,dirs_name,logfile,flg=0.5):
    for i in range(len(dirs)):
        outdir = pdir+dirs[i]+'*' 
        tmp = glob.glob(outdir)
        acc_ = tmp[0] + '/sup/acc_selective.npy'
        iterp = tmp[0] + '/sup/iter_p.npy'
        acc = np.load(acc_)
        iter_p = np.load(iterp)
        fin_acc = torch.tensor(acc[acc.shape[0]-1])
        num_task = fin_acc.size()[0]

        acc_forget = (fin_acc[:num_task-1,0]).mean()
        acc_prev = (fin_acc[:num_task-1,1]).mean()
        acc_new = flg*fin_acc[num_task-1,1] + (1-flg)*fin_acc[num_task-1,0]

        print(outdir)
        print(tmp[0])
        if i==0:
            with open(logfile, 'w') as f:
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i],acc_forget,acc_prev,acc_prev-acc_forget,acc_new ))
        else:
            with open(logfile, 'a') as f:
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i],acc_forget,acc_prev,acc_prev-acc_forget,acc_new ))

def cmp_Aaves_newdef(mat_acc, num_acc,tsk):
    s = mat_acc.size(0)
    aves = torch.zeros(s,2)
    tsk = int(tsk)
    for ss in range(s):
        mat_acc_fin = mat_acc[ss]
        mat_num_fin = num_acc[ss]

        mat_acc_keep = mat_acc_fin[:tsk,1]
        mat_num_keep = mat_num_fin[:tsk,1]
        mat_acc_del  = mat_acc_fin[:tsk,0]
        mat_num_del  = mat_num_fin[:tsk,0]


        A_ave     = ((mat_acc_keep*mat_num_keep).sum()  )/( (mat_num_keep).sum()  )
        A_ave_del = ((mat_acc_del*mat_num_del).sum() )/( (mat_num_del).sum()  )

        aves[ss,0] = A_ave_del
        aves[ss,1] = A_ave
    return aves


def cmp_Aaves(mat_acc, num_acc,tsk):
    s = mat_acc.size(0)
    aves = torch.zeros(s,2)
    tsk = int(tsk)
    for ss in range(s):
        mat_acc_fin = mat_acc[ss]
        mat_num_fin = num_acc[ss]

        if tsk >0:
            mat_acc_keep = mat_acc_fin[:tsk,1]
            mat_num_keep = mat_num_fin[:tsk,1]
            mat_acc_del  = mat_acc_fin[:tsk,0]
            mat_num_del  = mat_num_fin[:tsk,0]
        else:
            mat_acc_keep = torch.tensor([0])
            mat_num_keep = torch.tensor([0])
            mat_acc_del  = torch.tensor([0])
            mat_num_del  = torch.tensor([0])

        mat_acc_keep_fin = mat_acc_fin[tsk,1] 
        mat_num_keep_fin = mat_num_fin[tsk,1] 
        mat_acc_del_fin = mat_acc_fin[tsk,0] 
        mat_num_del_fin = mat_num_fin[tsk,0] 

        A_ave     = ((mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum()   )/( (mat_num_keep).sum()+(mat_num_keep_fin).sum()   )
        A_ave_del = ((mat_acc_del*mat_num_del).sum() + (mat_acc_del_fin*mat_num_del_fin).sum()   )/( (mat_num_del).sum()+(mat_num_del_fin).sum()   )

        aves[ss,0] = A_ave_del
        aves[ss,1] = A_ave
    return aves


def out_res_3(pdir,dirs,dirs_name,logfile, max_tsk=5, epoch=200):
    # Average Accuracy
    # Forgetting 

    # mat_acc = torch.tensor(acc_selective)
    # num_acc = torch.tensor(num_selective)
    tsk = max_tsk

    results = {}
    results_Akeeps = []
    results_Aaves = []
    results_Fkeeps = []
    results_Fdels = []
    outdirs=[]


    # ディレクトリごとに計算
    for i in range(len(dirs)):
        outdir = pdir+'*'+dirs[i]+'*' 
        tmp = glob.glob(outdir)

        # 対象のディレクトリが有るかを計算
        if len(tmp)>0:
            outdirs.append(dirs[i])
            iter_name = tmp[0] + '/sup/iter_p.npy'
            acc_selective_name = tmp[0] + '/sup/acc_selective.npy'
            num_selective_name = tmp[0] + '/sup/num_selective.npy'
            acc_selective = np.load(acc_selective_name)
            num_selective = np.load(num_selective_name)

            mat_acc = torch.tensor(acc_selective)
            num_acc = torch.tensor(num_selective)
            iters   = torch.tensor(np.load(iter_name))
            num_iters = len(iters)
            A_keeps= torch.zeros(num_iters)
            A_aves= torch.zeros(num_iters)
            F_keeps= torch.zeros(num_iters)
            F_dels= torch.zeros(num_iters)
            for itr in range(num_iters):
                tsk = min(int(((iters[itr].round()+0.001)/epoch).floor()),max_tsk)
                # そのエポックのタイミングで最後のタスクを用いてACCを計算
                # mac_acc_maxの定義を変えない場合
                # aves = cmp_Aaves(mat_acc, num_acc,tsk)
                # mac_acc_maxの定義を変える場合
                aves = cmp_Aaves_newdef(mat_acc, num_acc,tsk+1)

                mat_acc_max, _ = aves.max(0)

                mat_acc_fin = mat_acc[itr]
                mat_num_fin = num_acc[itr]

                mat_acc_keep = mat_acc_fin[:tsk,1]
                mat_num_keep = mat_num_fin[:tsk,1]
                mat_num_del  = mat_num_fin[:tsk,0]
                mat_acc_keep_fin = mat_acc_fin[tsk,1] 
                mat_num_keep_fin = mat_num_fin[tsk,1] 

                A_ave  = ((mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum()   )/( (mat_num_keep).sum()+(mat_num_keep_fin).sum()   )
                if tsk>0:
                    A_keep = ((mat_acc_keep*mat_num_keep).sum())/((mat_num_keep).sum())
                else:
                    A_keep = 0
                
                mat_F_keep = (mat_acc_max[1] - mat_acc_fin[:tsk,1])
                mat_F_del  = (mat_acc_max[0] - mat_acc_fin[:tsk,0])

                if tsk>0:
                    F_keep = ((mat_F_keep*mat_num_keep).sum())/((mat_num_keep).sum())
                    F_del  = ((mat_F_del*mat_num_del).sum())/((mat_num_del).sum())
                else:
                    F_keep = 0
                    F_del  = 0
                
                A_keeps[itr] = A_keep
                A_aves[itr]  = A_ave
                F_keeps[itr] = F_keep
                F_dels[itr]  = F_del
                print('{}\t{}\t{}\t{}\t{}\t{}'.format(itr, dirs_name[i], A_keep, A_ave, F_keep, F_del ))
                if itr==0:
                    with open('./res_acc/'+dirs_name[i]+logfile, 'w') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(itr, dirs_name[i], A_keep, A_ave, F_keep, F_del ))
                else:
                    with open('./res_acc/'+dirs_name[i]+logfile, 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(itr, dirs_name[i], A_keep, A_ave, F_keep, F_del ))

            print('{}\t{}\t{}\t{}\t{}'.format(dirs_name[i], A_keep, A_ave, F_keep, F_del ))
            dname_A_keeps = dirs_name[i]+'_A_keeps'
            dname_A_aves = dirs_name[i]+'_A_aves'
            dname_F_keeps = dirs_name[i]+'_F_keeps'
            dname_F_dels = dirs_name[i]+'_F_dels'
            results[dname_A_keeps] =A_keeps
            results[dname_A_aves]  =A_aves
            results[dname_F_keeps] =F_keeps
            results[dname_F_dels]  =F_dels
            results_Akeeps.append(A_keeps)
            results_Aaves.append(A_aves)
            results_Fkeeps.append(F_keeps)
            results_Fdels.append(F_dels)


            if i==0:
                with open('./res_acc/'+logfile, 'w') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, A_ave, F_keep, F_del ))
            else:
                with open('./res_acc/'+logfile, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, A_ave, F_keep, F_del ))

    # np.save(logfile+'.npz', F_del_all)
    num_itr = len(results_Akeeps[0])
    #-------------------
    with open('./res_acc/A_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Akeeps)):
            with open('./res_acc/A_keeps'+logfile, 'a') as f: f.write('{}\t'.format(results_Akeeps[k][i]))
        with open('./res_acc/A_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc/A_aves'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Aaves)):
            with open('./res_acc/A_aves'+logfile, 'a') as f: f.write('{}\t'.format(results_Aaves[k][i]))
        with open('./res_acc/A_aves'+logfile, 'a') as f: f.write('\n')
    #-------------------
                
    #-------------------
    with open('./res_acc/F_dels'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Fdels)):
            with open('./res_acc/F_dels'+logfile, 'a') as f: f.write('{}\t'.format(results_Fdels[k][i]))
        with open('./res_acc/F_dels'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc/F_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Fkeeps)):
            with open('./res_acc/F_keeps'+logfile, 'a') as f: f.write('{}\t'.format(results_Fkeeps[k][i]))
        with open('./res_acc/F_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------




    with open(logfile+'.npz',"wb") as f:
        pickle.dump(results, f)

    return  

def eval_H(logfile_in, epoch, task):

    f = open(logfile_in, 'r')
    datalist = f.readlines()
    H = []
    H2 = []
    for i in range(task):
        idx = (i+1)*epoch + (i+1) -1
        if len(datalist)>idx:
            datalist_tmp = datalist[idx]
            idx_H_ini=datalist_tmp.index('H:')+2
            idx_H_end=datalist_tmp.index(', H2')
            idx_H2_ini=datalist_tmp.index(', H2:')+5
            idx_H2_end=datalist_tmp.index('|\n')

            H_tmp = float(datalist_tmp[idx_H_ini:idx_H_end])
            H2_tmp = float(datalist_tmp[idx_H2_ini:idx_H2_end])
        else:
            H_tmp = 0
            H2_tmp = 0
        
        H.append(H_tmp)
        H2.append(H2_tmp)

    return H, H2


def out_res_lsf(pdir,dirs,dirs_name,logfile, max_tsk=5, epoch=200):
    # Average Accuracy
    # Forgetting 

    tsk = max_tsk

    results = {}
    results_Akeeps = []
    results_Aaves = []
    results_Fkeeps = []
    results_Fdels = []
    results_HOS = []
    outdirs=[]


    # ディレクトリごとに計算
    for i in range(len(dirs)):
        outdir = pdir+dirs[i]
        tmp = glob.glob(outdir)

        if len(tmp)==0: continue

        f = open(tmp[0] + '/log.txt', 'r')
        datalist = f.readlines()
        if ((max_tsk+1)*epoch + (max_tsk+1)) != len(datalist): 
            print('break:{}'.format(tmp[0]))
            continue

        # 対象のディレクトリが有るかを計算
        if len(tmp)>0:
            logfile_in = tmp[0] + '/log.txt'


            H, H2 = eval_H(logfile_in,epoch, (max_tsk+1))


            outdirs.append(dirs[i])
            iter_name = tmp[0] + '/sup/iter_p.npy'
            acc_selective_name = tmp[0] + '/sup/acc_selective.npy'
            num_selective_name = tmp[0] + '/sup/num_selective.npy'
            acc_selective = np.load(acc_selective_name)
            num_selective = np.load(num_selective_name)

            mat_acc = torch.tensor(acc_selective)
            num_acc = torch.tensor(num_selective)
            iters   = torch.tensor(np.load(iter_name))
            num_iters = len(iters)
            A_keeps= torch.zeros((max_tsk+1))
            A_aves= torch.zeros((max_tsk+1))
            F_keeps= torch.zeros((max_tsk+1))
            F_dels= torch.zeros((max_tsk+1))
            HOSs= torch.zeros((max_tsk+1))
            # A_keeps= torch.zeros(num_iters)
            # A_aves= torch.zeros(num_iters)
            # F_keeps= torch.zeros(num_iters)
            # F_dels= torch.zeros(num_iters)

            for itr in range((max_tsk+1)):
                idx = (itr+1)*epoch + (itr+1) -1
                # if itr != min(int(((iters[idx].round())/epoch).floor()),max_tsk): 
                #     print('error')
                #     break
                tsk = itr

                # そのエポックのタイミングで最後のタスクを用いてACCを計算
                # mac_acc_maxの定義を変えない場合
                # aves = cmp_Aaves(mat_acc, num_acc,tsk)
                # mac_acc_maxの定義を変える場合
                aves = cmp_Aaves_newdef(mat_acc, num_acc,tsk+1)

                mat_acc_max, _ = aves.max(0)

                mat_acc_fin = mat_acc[idx]
                mat_num_fin = num_acc[idx]

                mat_acc_keep = mat_acc_fin[:tsk,1]
                mat_num_keep = mat_num_fin[:tsk,1]
                mat_num_del  = mat_num_fin[:tsk,0]
                mat_acc_keep_fin = mat_acc_fin[tsk,1] 
                mat_num_keep_fin = mat_num_fin[tsk,1] 
                mat_acc_del_fin = mat_acc_fin[tsk,0] 
                mat_num_del_fin = mat_num_fin[tsk,0] 

                A_ave  = ( (mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum() + (mat_acc_del_fin*mat_num_del_fin).sum()   ) \
                    /( (mat_num_keep).sum()+(mat_num_keep_fin).sum() +(mat_num_del_fin).sum()   )
                # A_ave  = ((mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum()   )/( (mat_num_keep).sum()+(mat_num_keep_fin).sum()   )
                if tsk>0:
                    A_keep = ((mat_acc_keep*mat_num_keep).sum())/((mat_num_keep).sum())
                else:
                    A_keep = 0
                
                mat_F_keep = (mat_acc_max[1] - mat_acc_fin[:tsk,1])
                mat_F_del  = (mat_acc_max[0] - mat_acc_fin[:tsk,0])

                if tsk>0:
                    F_keep = ((mat_F_keep*mat_num_keep).sum())/((mat_num_keep).sum())
                    F_del  = ((mat_F_del*mat_num_del).sum())/((mat_num_del).sum())
                else:
                    F_keep = 0
                    F_del  = 0
                
                A_keeps[tsk-1] = A_keep
                A_aves[tsk-1]  = A_ave
                F_keeps[tsk-1] = F_keep
                F_dels[tsk-1]  = F_del

                HOS = (2*A_ave*F_del)/(A_ave+F_del+0.00001)
                HOSs[itr]  = HOS

                print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(idx, dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))
                if idx==0:
                    with open('./res_acc/'+dirs_name[i]+logfile, 'w') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))
                else:
                    with open('./res_acc/'+dirs_name[i]+logfile, 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))

            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))
            dname_A_keeps = dirs_name[i]+'_A_keeps'
            dname_A_aves = dirs_name[i]+'_A_aves'
            dname_F_keeps = dirs_name[i]+'_F_keeps'
            dname_F_dels = dirs_name[i]+'_F_dels'
            dname_HOS = dirs_name[i]+'_HOS'
            results[dname_A_keeps] =A_keeps
            results[dname_A_aves]  =A_aves
            results[dname_F_keeps] =F_keeps
            results[dname_F_dels]  =F_dels
            results[dname_HOS]  =HOS
            results_Akeeps.append(A_keeps)
            results_Aaves.append(A_aves)
            results_Fkeeps.append(F_keeps)
            results_Fdels.append(F_dels)
            results_HOS.append(HOSs)

            if i==0:
                with open('./res_acc/'+logfile, 'w') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))
            else:
                with open('./res_acc/'+logfile, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS, H[tsk], H2[tsk] ))

    # np.save(logfile+'.npz', F_del_all)
    num_itr = len(results_Akeeps[0])
    #-------------------
    with open('./res_acc/A_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Akeeps)):
            with open('./res_acc/A_keeps'+logfile, 'a') as f: f.write('{}\t'.format(results_Akeeps[k][i]))
        with open('./res_acc/A_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc/A_aves'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Aaves)):
            with open('./res_acc/A_aves'+logfile, 'a') as f: f.write('{}\t'.format(results_Aaves[k][i]))
        with open('./res_acc/A_aves'+logfile, 'a') as f: f.write('\n')
    #-------------------
                
    #-------------------
    with open('./res_acc/F_dels'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Fdels)):
            with open('./res_acc/F_dels'+logfile, 'a') as f: f.write('{}\t'.format(results_Fdels[k][i]))
        with open('./res_acc/F_dels'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc/F_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        for k in range(len(results_Fkeeps)):
            with open('./res_acc/F_keeps'+logfile, 'a') as f: f.write('{}\t'.format(results_Fkeeps[k][i]))
        with open('./res_acc/F_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------




    with open(logfile+'.npz',"wb") as f:
        pickle.dump(results, f)

    return  

if __name__ == '__main__': # 00_CIFAR_INC_Bench_R03_1023 DIGIT_Bench

    # if not os.path.exists('./res_acc/'): os.mkdir('./res_acc/')
    # # #------------------------------------------------
    # logfile = '[LSF-ICASSP-r51]-CIFAR-T2.txt'
    # pdir='./result/*LSF-ICASSP-r51*-CIFAR-T2/'
    # tsk = 1
    # dirs           = [  'R0-KEP0-LMD0-E1_*', 'R0-KEP0-LMD0-E10_*', 'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*', 'R0-KEP10-LMD10-E1_*', 'R0-KEP10-LMD10-E10_*', 'R0-KEP10-LMD100-E1_*', 'R0-KEP10-LMD100-E10_*',\
    #                     'R1-KEP0-LMD0-E1_*', 'R1-KEP0-LMD0-E10_*', 'R1-KEP10-LMD0-E1_*', 'R1-KEP10-LMD0-E10_*', 'R1-KEP10-LMD10-E1_*', 'R1-KEP10-LMD10-E10_*', 'R1-KEP10-LMD100-E1_*', 'R1-KEP10-LMD100-E10_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = '[LSF-ICASSP-r61]-CIFAR-T2.txt'
    # pdir='./result/*LSF-ICASSP-r61*-CIFAR-T2/'
    # tsk = 1
    # dirs           = [  'R0-KEP0-LMD0-E1_*', 'R0-KEP0-LMD0-E10_*', 'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*', 'R0-KEP10-LMD10-E1_*', 'R0-KEP10-LMD10-E10_*', 'R0-KEP10-LMD100-E1_*', 'R0-KEP10-LMD100-E10_*',\
    #                     'R1-KEP0-LMD0-E1_*', 'R1-KEP0-LMD0-E10_*', 'R1-KEP10-LMD0-E1_*', 'R1-KEP10-LMD0-E10_*', 'R1-KEP10-LMD10-E1_*', 'R1-KEP10-LMD10-E10_*', 'R1-KEP10-LMD100-E1_*', 'R1-KEP10-LMD100-E10_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = '[LSF-ICASSP-r5-rev2]-CIFAR-T5.txt'
    # pdir='./result/*LSF-ICASSP-r5-rev2*-CIFAR-T5/'
    # tsk = 4
    # # dirs           = [  'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*']
    # dirs           = [  'R0-KEP0-LMD0-E1_*', 'R0-KEP0-LMD0-E10_*', 'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*', 'R0-KEP10-LMD10-E1_*', 'R0-KEP10-LMD10-E10_*', 'R0-KEP10-LMD100-E1_*', 'R0-KEP10-LMD100-E10_*',\
    #                     'R1-KEP0-LMD0-E1_*', 'R1-KEP0-LMD0-E10_*', 'R1-KEP10-LMD0-E1_*', 'R1-KEP10-LMD0-E10_*', 'R1-KEP10-LMD10-E1_*', 'R1-KEP10-LMD10-E10_*', 'R1-KEP10-LMD100-E1_*', 'R1-KEP10-LMD100-E10_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    logfile = '[LSF-ICASSP-r6-rev2]-CIFAR-T5.txt'
    pdir='./result/*LSF-ICASSP-r6-rev2*-CIFAR-T5/'
    tsk = 4
    # dirs           = [  'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*']
    dirs           = [  'R1-KEP0-LMD0-E1-A0*', 'R0-KEP0-LMD0-E1_*', 'R0-KEP0-LMD0-E10_*', 'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*', 'R0-KEP10-LMD10-E1_*', 'R0-KEP10-LMD10-E10_*', 'R0-KEP10-LMD100-E1_*', 'R0-KEP10-LMD100-E10_*',\
                        'R1-KEP0-LMD0-E1_*', 'R1-KEP0-LMD0-E10_*', 'R1-KEP10-LMD0-E1_*', 'R1-KEP10-LMD0-E10_*', 'R1-KEP10-LMD10-E1_*', 'R1-KEP10-LMD10-E10_*', 'R1-KEP10-LMD100-E1_*', 'R1-KEP10-LMD100-E10_*']
    dirs_name      = dirs
    out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CIFAR-T2-0720.txt'
    # pdir='./result/99-LSF-CIFAR-T2-0720/'
    # tsk = 1
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CIFAR-T5-0720.txt'
    # pdir='./result/99-LSF-CIFAR-T5-0720/'
    # tsk = 4
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CIFAR-T2-INC-0720.txt'
    # pdir='./result/99-LSF-CIFAR-T2-INC-0720/'
    # tsk = 1
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CIFAR-T5-INC-0720.txt'
    # pdir='./result/99-LSF-CIFAR-T5-INC-0720/'
    # tsk = 4
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CIFAR-T10-0720.txt'
    # pdir='./result/99-LSF-CIFAR-T10-0720/'
    # tsk = 9
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-CUB-T5-0720.txt'
    # pdir='./result/99-LSF-CUB-T5-0720/'
    # tsk = 4
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)

    # #------------------------------------------------
    # logfile = 'LSF-STN-T4.txt'
    # pdir='./result/99-LSF-STN-T4/'
    # tsk = 3
    # dirs           = [ '00_*','01_*','02_*','03_*','04_*','05_*','06_*','07_*']
    # dirs_name      = dirs
    # out_res_lsf(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)



    # logfile = '_00_CIFAR_Bench_T10.txt'
    # pdir='./result/CIFAR_Bench_T10/'
    # tsk = 10
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+']
    # # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=9, epoch=200)

    # logfile = '_00_CIFAR_Bench.txt'
    # pdir='./result/00_CIFAR_Bench/'
    # tsk = 2
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=200)

    # logfile = '_00_CIFAR_Bench_T5.txt'
    # pdir='./result/CIFAR_Bench_T5/'
    # tsk = 5
    # # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_']
    # # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+']
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=200)


    # logfile = '_DIGIT_Bench_perm5.txt'
    # pdir='./result/DIGIT_Bench/PERM5/'
    # tsk = 5
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=30)

    # logfile = '_DIGIT_Bench_perm5.txt'
    # pdir='./result/DIGIT_Bench/PERM5/'
    # tsk = 5
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=30)

    # logfile = '_DIGIT_Bench_perm.txt'
    # pdir='./result/DIGIT_Bench/PERM/'
    # tsk = 3
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=2, epoch=30)

    # logfile = '_MNIST_SVHN.txt'
    # pdir='./result/DIGIT_Bench/MNIST_SVHN/'
    # tsk = 2
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=50)

    # logfile = '_SVHN_MNIST.txt'
    # pdir='./result/DIGIT_Bench/SVHN_MNIST/'
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '09_Prop_', '10_Prop_', '11_MAS+']
    # dirs_name      = [ '00_Base*', '01_LwF', '02_LwF+', '03_EWC', '04_EWC+', '05_EWC+_LwF+', '06_MAS', '08_MAS+', '09_Prop_LwF+', '10_Prop_EwC+', '11_Prop_MAS+']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=50)

    # logfile = '_00_CIFAR_INC_Bench_R03_1023.txt'
    # pdir='./result/00_CIFAR_INC_Bench_R03_1023/'
    # tsk = 2
    # dirs      = [ '00_Base*', '01_LwF_', '02_LwFS_', '03_EWC_', '04_EWCS_', '05_EWCS_LwFS_', '06_EWC_LwF_', '07_Prop_LwF_', '08_Prop_LwFS_', '09_Prop_EWC_', '10_Prop_EWCS_','11_Prop_','12_Prop_','13_MAS_','14_MASS_','15_Prop_']
    # dirs_name = [ '00_Base*', '01_LwF_', '02_LwFS_', '03_EWC_', '04_EWCS_', '05_EWCS_LwFS_', '06_EWC_LwF_', '07_Prop_LwF_', '08_Prop_LwFS_', '09_Prop_EWC_', '10_Prop_EWCS_','11_Prop_','12_Prop_','13_MAS_','14_MASS_','15_Prop_']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=200)

    # logfile = 'DIGIT_Bench_1021_svhn.txt'
    # pdir='./result/DIGIT_Bench_1021/SVHN_MNIST/'
    # dirs      = [ '00_Base*', '01_LwF_', '02_LwFS_', '03_EWC_', '04_EWCS_', '05_EWCS_LwFS_', '06_EWC_LwF_', '07_Prop_LwF_', '08_Prop_LwFS_', '09_Prop_EWC_', '10_Prop_EWCS_','11_Prop_']
    # dirs_name = [ '00_Base*', '01_LwF_', '02_LwFS_', '03_EWC_', '04_EWCS_', '05_EWCS_LwFS_', '06_EWC_LwF_', '07_Prop_LwF_', '08_Prop_LwFS_', '09_Prop_EWC_', '10_Prop_EWCS_','11_Prop_']
    # out_res(pdir,dirs,dirs_name,logfile)




