from argparse import ArgumentParser
import numpy as np
import torch
import datetime
import os
import subprocess
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


def out_res_3(pdir,dirs,dirs_name,logfile, max_tsk=5, epoch=200, epoch_table=[201,402,603,804,1004]):
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
    results_HOS = []

    outdirs=[]
    ii=0

    # ディレクトリごとに計算
    for i in range(len(dirs)):
        outdir = pdir+dirs[i]+'*' 
        # outdir = pdir+'*'+dirs[i]+'*' 
        tmp = glob.glob(outdir)

        # 対象のディレクトリが有るかを計算
        if len(tmp)>0:
            for l in range(len(tmp)):
                outdirs.append(dirs[i])
                iter_name = tmp[l] + '/sup/iter_p.npy'
                acc_selective_name = tmp[l] + '/sup/acc_selective.npy'
                num_selective_name = tmp[l] + '/sup/num_selective.npy'
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
                HOSs= torch.zeros(num_iters)

                if num_iters >= epoch_table[max_tsk]:

                    for itr in range(num_iters):
                        tsk = min(int(((iters[itr].round()-0.001)/epoch).floor()),max_tsk)
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
                        mat_acc_del_fin = mat_acc_fin[tsk,0] 
                        mat_num_del_fin = mat_num_fin[tsk,0] 

                        A_ave  = ( (mat_acc_keep*mat_num_keep).sum() + (mat_acc_keep_fin*mat_num_keep_fin).sum() + (mat_acc_del_fin*mat_num_del_fin).sum()   )/( (mat_num_keep).sum()+(mat_num_keep_fin).sum() +(mat_num_del_fin).sum()   )
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
                        
                        A_keeps[itr] = A_keep
                        A_aves[itr]  = A_ave
                        F_keeps[itr] = F_keep
                        F_dels[itr]  = F_del

                        HOS = (2*A_ave*F_del)/(A_ave+F_del+0.00001)
                        HOSs[itr]  = HOS
                        # print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(int(iters[itr].round()), dirs_name[i], A_keep, A_ave, F_keep, F_del, HOS ))
                        if itr==0:
                            with open('./res_acc_ijcai_0118/'+dirs_name[i]+logfile, 'w') as f:
                                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(int(iters[itr].round()), dirs_name[i], A_keep, A_ave, F_keep, F_del, HOS ))
                        else:
                            with open('./res_acc_ijcai_0118/'+dirs_name[i]+logfile, 'a') as f:
                                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(int(iters[itr].round()), dirs_name[i], A_keep, A_ave, F_keep, F_del, HOS ))

                    print('{}\t{}\t{}\t{}\t{}\t{}'.format(dirs_name[i], A_keep, A_ave, F_keep, F_del, HOS ))
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


                    if ii==0:
                        ii=1
                        with open('./res_acc_ijcai_0118/'+logfile, 'w') as f:
                            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, F_keep,  A_ave, F_del, HOS ))
                    else:
                        with open('./res_acc_ijcai_0118/'+logfile, 'a') as f:
                            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(dirs_name[i], A_keep, F_keep, A_ave, F_del, HOS ))

    # np.save(logfile+'.npz', F_del_all)
    num_itr = len(results_Akeeps[0])
    #-------------------
    with open('./res_acc_ijcai_0118/HOS'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        if i in epoch_table:
            for k in range(len(results_HOS)):
                with open('./res_acc_ijcai_0118/HOS'+logfile, 'a') as f:
                    if i <= len(results_HOS[k]): 
                        f.write('{}\t'.format(results_HOS[k][i]))
                    else:
                        f.write('--\t')
            with open('./res_acc_ijcai_0118/HOS'+logfile, 'a') as f: f.write('\n')

    with open('./res_acc_ijcai_0118/A_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        if i in epoch_table:
            for k in range(len(results_Akeeps)):
                with open('./res_acc_ijcai_0118/A_keeps'+logfile, 'a') as f: 
                    if i <= len(results_Akeeps[k]): 
                        f.write('{}\t'.format(results_Akeeps[k][i]))
                    else:
                        f.write('--\t')
            with open('./res_acc_ijcai_0118/A_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc_ijcai_0118/A_aves'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        if i in epoch_table:
            for k in range(len(results_Aaves)):
                with open('./res_acc_ijcai_0118/A_aves'+logfile, 'a') as f: 
                    if i <= len(results_Aaves[k]): 
                        f.write('{}\t'.format(results_Aaves[k][i]))
                    else:
                        f.write('--\t')
            with open('./res_acc_ijcai_0118/A_aves'+logfile, 'a') as f: f.write('\n')
    #-------------------
                
    #-------------------
    with open('./res_acc_ijcai_0118/F_dels'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        if i in epoch_table:
            for k in range(len(results_Fdels)):
                with open('./res_acc_ijcai_0118/F_dels'+logfile, 'a') as f: 
                    if i <= len(results_Fdels[k]): 
                        f.write('{}\t'.format(results_Fdels[k][i]))
                    else:
                        f.write('--\t')
            with open('./res_acc_ijcai_0118/F_dels'+logfile, 'a') as f: f.write('\n')
    #-------------------

    #-------------------
    with open('./res_acc_ijcai_0118/F_keeps'+logfile, 'w') as f: 
        for i in range(len(outdirs)): f.write('{}\t'.format(outdirs[i]))
        f.write('\n')
    for i in range(num_itr):
        if i in epoch_table:
            for k in range(len(results_Fkeeps)):
                with open('./res_acc_ijcai_0118/F_keeps'+logfile, 'a') as f: 
                    if i <= len(results_Fkeeps[k]): 
                        f.write('{}\t'.format(results_Fkeeps[k][i]))
                    else:
                        f.write('--\t')
            with open('./res_acc_ijcai_0118/F_keeps'+logfile, 'a') as f: f.write('\n')
    #-------------------

    with open(logfile+'.npz',"wb") as f:
        pickle.dump(results, f)

    return  

if __name__ == '__main__': # 00_CIFAR_INC_Bench_R03_1023 DIGIT_Bench

    #------------------------------------------------
    logfile = '[LSF-ICASSP-r5-rev2]-CIFAR-T5.txt'
    pdir='./result/*LSF-ICASSP-r5-rev2*-CIFAR-T5/'
    tsk = 4
    # dirs           = [  'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*']
    dirs           = [  'R0-KEP10-LMD0-E1_*', 'R0-KEP10-LMD0-E10_*', 'R0-KEP10-LMD10-E1_*', 'R0-KEP10-LMD10-E10_*',\
                        'R1-KEP10-LMD0-E1_*', 'R1-KEP10-LMD0-E10_*', 'R1-KEP10-LMD10-E1_*', 'R1-KEP10-LMD10-E10_*']
    dirs_name      = dirs
    out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=tsk, epoch=200)


    # logfile = '_02_CUB_T5_Bench.txt'
    # pdir='./result-ijcai21/02_CUB_T5_Bench/'
    # tsk = 5
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # dirs_name      = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=200, epoch_table=[200,401,602,803,1004])

    # logfile = '_02_STN_T5_Bench.txt'
    # pdir='./result-ijcai21/02_STN_T5_Bench/'
    # tsk = 4
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # dirs_name      = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=3, epoch=200, epoch_table=[200,401,602,803,1004])

    # logfile = '_00_CIFAR_Bench_T5.txt'
    # pdir='./result-ijcai21/CIFAR_Bench_T5/'
    # tsk = 5
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # dirs_name      = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=200, epoch_table=[200,401,602,803,1004,1205,1406,1607,1808,2009])

    # logfile = '_00_CIFAR_Bench.txt'
    # pdir='./result-ijcai21/00_CIFAR_Bench/'
    # tsk = 2
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # dirs_name      = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=200, epoch_table=[200,401])

    # logfile = '_00_CIFAR_Bench_T10.txt'
    # pdir='./result-ijcai21/CIFAR_Bench_T10/'
    # tsk = 10
    # dirs           = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # dirs_name      = [ '00_Base*', '01_LwF_', '02_LwF+_', '03_EWC_', '04_EWC+_', '05_EWC+_', '06_MAS_', '08_MAS+_', '10_Prop_*K10', '11_MAS+*K10' ]
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=9, epoch=200, epoch_table=[200,401,602,803,1004,1205,1406,1607,1808,2009])

    # #-------------------------------------
    # logfile = '_00_CIFAR_Bench_T5_Resize.txt'
    # pdir='./result-ijcai21/CIFAR_Bench_T5_Resize/'
    # tsk = 5
    # dirs           = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand2_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # dirs_name      = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand2_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=200, epoch_table=[200,401,602,803,1004,1205,1406,1607,1808,2009])

    # logfile = '_00_CIFAR_Bench_Resize.txt'
    # pdir='./result-ijcai21/00_CIFAR_Bench_Rsize/'
    # tsk = 2
    # dirs           = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand2_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # dirs_name      = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand2_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=1, epoch=200, epoch_table=[200,401,602,803,1004,1205,1406,1607,1808,2009])

    # logfile = '_02_CUB_T5_Resize.txt'
    # pdir='./result-ijcai21/02_CUB_T5_Bench_Resize/'
    # tsk = 5
    # dirs           = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand4_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand64_*']
    # dirs_name      = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand4_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand64_*']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=4, epoch=200, epoch_table=[200,401,602,803,1004])

    # logfile = '_02_STN_T5_Resize.txt'
    # pdir='./result-ijcai21/02_STN_T5_Bench_Resize/'
    # tsk = 4
    # dirs           = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand4_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # dirs_name      = [ '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand1_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand4_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand8_', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand16_*', '10_Prop_EWC+_LwF+_LF5_K10_EW100_Rand32_*']
    # out_res_3(pdir,dirs,dirs_name,logfile,max_tsk=3, epoch=200, epoch_table=[200,401,602,803,1004])
