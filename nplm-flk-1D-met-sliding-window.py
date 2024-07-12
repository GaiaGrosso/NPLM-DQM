import math, time, datetime, h5py, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from joblib import dump,load
import os as os
from FLKutils import *
from SampleUtils import *
# shape only
# lumiS 250 -- 500
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--toys', type=int, help="toys", required=False, default=100)
parser.add_argument('-w', '--widthperc', type=int, help="k nearest neighbors", required=False, default=90)
#parser.add_argument('-seed', '--seed', type=int, help="first seed", required=False, default=0)
args = parser.parse_args()

M = 1000
lam = 1e-6
iterations=10000000
flk_sigma_perc = args.widthperc        # flk width quantile pair-distance                                                                                          
Nexp = args.toys     # Number of pseudo-experiment

folder_root = "/n/holystore01/LABS/iaifi_lab/Lab/CMS-DQM/" # where the data are stored
vars_monitoring = ['PFMET_pt']
data_path = folder_root + 'MET.h5'
binsrange = {'PFMET_pt': np.linspace(0, 50, 40)}
yrange = {'PFMET_pt': [0, 20]}
# output                                                                                                                                                      
folder_out = '/n/home00/ggrosso/CMS-DQM/out/MET/sliding-window/1D/'
NP = 'M%i_lam%s_iter%i/'%(M, str(lam), iterations)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)

# Read samples
f = h5py.File(data_path, 'r')
lumi_all = np.array(f.get('lumi'))
data_all = np.array([])
for var in vars_monitoring:
    if var==vars_monitoring[0]:
        data_all = np.array(f[var]).reshape((-1, 1))
    else:
        data_all = np.concatenate((data_all, np.array(f[var]).reshape((-1, 1))), axis=1)
f.close()
print(data_all.shape)

lumiS_ordering_index = np.argsort(lumi_all)
data_all = data_all[lumiS_ordering_index]

random_seeds=np.random.randint(0, 100000, Nexp)
pval_list = []
for sw in np.arange(600, 700, 1):
    tnull = np.array([])
    tobs  = 0
    ref_sw = data_all[(lumi_all>=(sw-10))*(lumi_all<=(sw-1))]
    data_sw = data_all[lumi_all==(sw)]
    Nd = data_sw.shape[0]
    Nr = ref_sw.shape[0]
    w_ref = Nd*1./Nr
    mean_all, std_all = np.mean(ref_sw, axis=0), np.std(ref_sw, axis=0)
    ref_sw  = standardize(ref_sw, mean_all, std_all).astype('f')
    data_sw = standardize(data_sw, mean_all, std_all).astype('f')
    input_sw = np.concatenate((ref_sw, data_sw), axis=0)
    #target
    label_R = np.zeros((Nr,)).astype('f')
    label_D = np.ones((Nd,)).astype('f')
    labels  = np.concatenate((label_D,label_R), axis=0).reshape((-1, 1))

    # candidate sigma                                                                                                                                       
    flk_sigma = candidate_sigma(ref_sw[:1000, :], perc=flk_sigma_perc)                                                                                 
    print('flk_sigma', flk_sigma)
    #if os.path.exists('%s/tvalues_flk-sigma%s_%i.h5'%(folder_out+NP, flk_sigma, sw)):
    #    print('Skeep LumiS ', sw)
    #    continue
    # calibrate
    for i in np.arange(Nexp):
        seed = int(random_seeds[i]+sw)
        rng = np.random.default_rng(seed=seed)
        index = np.arange(Nd+Nr)
        rng.shuffle(index)
        input_tmp = input_sw[index]
        labels_tmp = labels
        plot_reco=False
        verbose=False

        flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
        t_tmp, pred_tmp = run_toy('toy%i'%(i), input_tmp, labels_tmp, w_ref, flk_config, seed,
                                  plot=plot_reco, verbose=verbose, savefig=plot_reco, output_path='./')
        tnull = np.append(tnull, t_tmp)
    plot_reco=True
    verbose=True
    # tobs
    seed = int(sw)
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    tobs, pred_obs = run_toy('tobs%i'%(sw), input_sw, labels, w_ref, flk_config, seed,
                             plot=plot_reco, verbose=verbose, savefig=plot_reco, output_path=folder_out+NP,
                             binsrange=binsrange, yrange=yrange, xlabels=vars_monitoring,
    )
    # pval
    pval = np.sum(tnull>=tobs)*1./len(tnull)
    
    # save stuff
    f_pvalue=open('%s/pval.txt'%(folder_out+NP), 'a')
    f_pvalue.write('%i,%f\n'%(sw,pval))
    f_pvalue.close()

    f = h5py.File('%s/tvalues_flk-sigma%s_%i.h5'%(folder_out+NP, flk_sigma, sw), 'w')
    f.create_dataset('tnull', data=tnull, compression='gzip')
    f.create_dataset('tobs', data=np.array([tobs]), compression='gzip')
    f.close()

    # print stuff
    print('lumiS: %i\np-val = '%(sw), pval)
    print('t_obs = ', tobs)
    print('<t_null> = ', np.mean(tnull))
