from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)


import numpy as np
import os, time
import torch
#UTILS

def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.
    # use on a (small) sample of reference data (standardize first if necessary)
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),1)

'''
def NP2_gen(size, seed):
    # custom function to generate samples of non-resonant new physics events
    if size>10000:
        raise Warning('Sample size is grater than 1000: Generator will not approximate the tail well')
    sample = np.array([])
    #normalization factor
    np.random.seed(seed)
    Norm = 256.*0.25*0.25*np.exp(-2)
    while(len(sample)<size):
        x = np.random.uniform(0,1) #assuming not to generate more than 10 000 events
        p = np.random.uniform(0, Norm)
        if p<= 256.*x*x*np.exp(-8.*x):
            sample = np.append(sample, x)
    return sample
'''

class non_res(rv_continuous):
    def _pdf(self, x):
        return 256 * (x**2) * np.exp(- 8 * x)

def nonres_sig(N_S, seed):
    # this function can be used to generate non-resonant signal events.
    my_sig = non_res(momtype = 0, a=0, b=1, seed=seed)
    sig_sample = my_sig.rvs(size = N_S)
    return sig_sample


def get_logflk_config(M,flk_sigma,lam,weight,iter=[1000000],seed=None,cpu=False):
    # it returns logfalkon parameters
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,
            'penalty_list' : lam, # list of regularization parameters,
            'iter_list' : iter, #list of number of CG iterations,
            'options' : FalkonOptions(cg_tolerance=np.sqrt(float(1e-7)), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }


def compute_t(preds,Y,weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))

def trainer(X,Y,flk_config):
    # trainer for logfalkon model
    Xtorch=torch.from_numpy(X)
    Ytorch=torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()
'''
def standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            vec = vec- mean
            vec = vec *1./ std
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.                                                                                                            \                                                                                                                                                                                             
            vec = vec *1./ mean
        dataset_new[:, j] = vec
    return dataset_new

def standardize(X):
    # standardize data as in HIGGS and SUSY
    for j in range(X.shape[1]):
        column = X[:, j]
        mean = np.mean(column)
        std = np.std(column)
        if np.min(column) < 0:
            column = (column-mean)*1./ std
        elif np.max(column) > 1.0:
            column = column *1./ mean
        X[:, j] = column

    return X
'''
def return_best_chi2dof(tobs):
    """
    Returns the most fitting value for dof assuming tobs follows a chi2_dof distribution,
    computed with a Kolmogorov-Smirnov test, removing NANs and negative values.
    Parameters
    ----------
    tobs : np.ndarray
        observations
    Returns
    -------
        best : tuple
            tuple with best dof and corresponding chi2 test result
    """

    dof_range = np.arange(np.nanmedian(tobs) - 10, np.nanmedian(tobs) + 10, 0.1)
    ks_tests = []
    for dof in dof_range:
        test = kstest(tobs, lambda x:chi2.cdf(x, df=dof))[0]
        ks_tests.append((dof, test))
    ks_tests = [test for test in ks_tests if test[1] != 'nan'] # remove nans
    ks_tests = [test for test in ks_tests if test[0] >= 0] # retain only positive dof
    best = min(ks_tests, key = lambda t: t[1]) # select best dof according to KS test result

    return best

def emp_zscore(t0,t1):
    if max(t0) <= t1:
        p_obs = 1 / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs
    else:
        p_obs = np.count_nonzero(t0 >= t1) / len(t0)
        Z_obs = round(norm.ppf(1 - p_obs),2)
        return Z_obs

def chi2_zscore(t1, dof):
    p = chi2.cdf(float('inf'),dof)-chi2.cdf(t1,dof)
    return norm.ppf(1 - p)


# PLOT UTILS
def plot_reconstruction(data, weight_data, ref, weight_ref, ref_preds, xlabels=[], yrange=None, binsrange=None,
                        save=False, save_path='', file_name=''):
    '''                                                                                                                                                                 
    Reconstruction of the data distribution learnt by the model.                                                                                                        
                                                                                                                                                                        
    df:              (int) chi2 degrees of freedom                                                                                                                      
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)                                                                           
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)                                                                             
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)                                                                      
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample                                                                                       
    tau_OBS:         (float) value of the tau term after training                                                                                                       
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training                                                      
    feature_labels:  (list of string) list of names of the training variables                                                                                           
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)                                                        
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)                                                              
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)                  
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)                                                  
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training (if not given, only tau reconstruction is plotted)
    '''
    # used to regularize empty reference bins                                                                                                                           
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data

    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        bins = np.linspace(np.min(ref[:, i]),np.max(ref[:, i]),50)
        if not binsrange==None:
            if len(binsrange[xlabels[i]]):
                bins=binsrange[xlabels[i]]
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.15, 0.43, 0.8, 0.5])
        hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = plt.hist(ref[:, i], weights=np.exp(ref_preds[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)

        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='RECO', color='#b2df8a', lw=1, s=30, zorder=4)

        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.15, 0.1, 0.8, 0.3])
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black')
        plt.plot(x, hN[0]/(hR[0]+eps), label ='RECO', color='#b2df8a', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        
        if len(xlabels)>0:
            plt.xlabel(xlabels[i], fontsize=22, fontname='serif')
        else:
            plt.xlabel('x', fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')

        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        #plt.ylim(0,10)
        if len(xlabels):
            if not yrange==None and len(xlabels)>0:
                plt.ylim(yrange[xlabels[i]][0], yrange[xlabels[i]][1]) 
        plt.grid()
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path+file_name.replace('.pdf', '_%i.pdf'%(i)))
            fig.savefig(save_path+file_name.replace('.pdf', '_%i.png'%(i)))
        plt.show()
        plt.close()
    return

def plot_reconstruction_df(df, data, weight_data, ref, weight_ref, t_obs, ref_preds,
                        save=False, save_path='', file_name=''):
    '''
    Reconstruction of the data distribution learnt by the model.

    df:              (int) chi2 degrees of freedom
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample
    tau_OBS:         (float) value of the tau term after training
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training
    feature_labels:  (list of string) list of names of the training variables
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training (if not given, only tau reconstruction is plotted)
    '''
    # used to regularize empty reference bins
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data


    Zscore=norm.ppf(chi2.cdf(t_obs, df))


    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
      bins = np.linspace(np.min(ref[:, i]),np.max(ref[:, i]),30)
      fig = plt.figure(figsize=(8, 8))
      fig.patch.set_facecolor('white')
      ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])
      hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
      hR = plt.hist(ref[:, i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
      hN = plt.hist(ref[:, i], weights=np.exp(ref_preds[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)

      plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
      plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='RECO', color='#b2df8a', lw=1, s=30, zorder=4)

      font = font_manager.FontProperties(family='serif', size=16)
      l    = plt.legend(fontsize=18, prop=font, ncol=2)
      font = font_manager.FontProperties(family='serif', size=18)
      title  = 't='+str(np.around(t_obs, 2))

      title += ', Z-score='+str(np.around(Zscore, 2))
      l.set_title(title=title, prop=font)
      plt.tick_params(axis='x', which='both',    labelbottom=False)
      plt.yticks(fontsize=16, fontname='serif')
      plt.xlim(bins[0], bins[-1])
      plt.ylabel("events", fontsize=22, fontname='serif')
      plt.yscale('log')
      ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3])
      x   = 0.5*(bins[1:]+bins[:-1])
      plt.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black')
      plt.plot(x, hN[0]/(hR[0]+eps), label ='RECO', color='#b2df8a', lw=3)

      font = font_manager.FontProperties(family='serif', size=16)
      plt.legend(fontsize=18, prop=font)
      plt.xlabel('x', fontsize=22, fontname='serif')
      plt.ylabel("ratio", fontsize=22, fontname='serif')

      plt.yticks(fontsize=16, fontname='serif')
      plt.xticks(fontsize=16, fontname='serif')
      plt.xlim(bins[0], bins[-1])
      #plt.ylim(0,10)
      plt.grid()
      if save:
          os.makedirs(save_path, exist_ok=True)
          fig.savefig(save_path+file_name)
      plt.show()
      plt.close()

    return

def err_bar(hist, n_samples):
    bins_counts = hist[0]
    bins_limits = hist[1]
    x   = 0.5*(bins_limits[1:] + bins_limits[:-1])
    bins_width = 0.5*(bins_limits[1:] - bins_limits[:-1])
    err = np.sqrt(np.array(bins_counts)/(n_samples*np.array(bins_width)))

    return x, err

def plot_data(data, label, name=None, dof=None, out_path=None, title=None,
                 density=True, bins=10,
                 c='mediumseagreen', e='darkgreen'):
    """
    Plot reference vs new physics t distribution
    Parameters
    ----------
    data : np.ndarray or list
        (N_toy,) array of observed test statistics
    dof : int
        degrees of freedom of the chi-squared distribution
    name : string
        filename for the saved figure
    out_path : string, optional
        output path where the figure will be saved. The default is ./fig.
    title : string
        title of the plot
    density : boolean
        True to normalize the histogram, false otherwise.
    bins : int or string, optional
        bins for the function plt.hist(). The default is 'fd'.
    Returns
    -------
    plot
    """
    plt.figure(figsize=(10,7))
    plt.style.use('classic')

    hist = plt.hist(data, bins = bins, color=c, edgecolor=e,
                        density=density, label = str(label))
    x_err, err = err_bar(hist, data.shape[0])
    plt.errorbar(x_err, hist[0], yerr = err, color=e, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)

    plt.ylim(bottom=0)

    # results data
    md_t = round(np.median(data), 2)
    if dof:
        z_chi2 = round(chi2_zscore(np.median(data),dof=dof),2)
    if dof:
        res = "md t = {} \nZ_chi2 = {}".format(md_t,z_chi2)
    else:
        res = "md t = {}".format(md_t)
    # plot chi2 and set xlim
    if dof:
        chi2_range = chi2.ppf(q=[0.00001,0.999], df=dof)
        x = np.arange(chi2_range[0], chi2_range[1], .05)
        chisq = chi2.pdf(x, df=dof)
        plt.plot(x, chisq, color='#d7191c', lw=2, label='$\chi^2(${}$)$'.format(dof))
        xlim = (min(chi2_range[0], min(data)-5), max(chi2_range[1], max(data)+5))
        plt.xlim(chi2_range)
    else:
        xlim = (min(data)-5, max(data)+5)
        plt.xlim(xlim)

    if title:
        plt.title(title, fontsize=20)

    plt.ylabel('P(t)', fontsize=20)
    plt.xlabel('t', fontsize=20)

    # Axes ticks
    ax = plt.gca()
    plt.legend(loc ="upper right", frameon=True, fontsize=18)
    ax.text(0.75, 0.65, res, color='black', fontsize=12,
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.5'),transform = ax.transAxes)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path+"/data_{}.pdf".format(name), bbox_inches='tight')
    plt.show()
    plt.close()


def plot_ref_data(ref, data, name=None, dof=None, out_path=None, title=None,
                 density=True, bins=10,
                 c_ref='#abd9e9', e_ref='#2c7bb6', c_sig='#fdae61', e_sig='#d7191c'):
    """
    Plot reference vs new physics t distribution
    Parameters
    ----------
    T_ref : np.ndarray or list
        (N_toy,) array of observed test statistics in the reference hypothesis
    T_sig : np.ndarray or list
        (N_toy,) array of observed test statistics in the New Physics hypothesis
    dof : int
        degrees of freedom of the chi-squared distribution
    name : string
        filename for the saved figure
    out_path : string, optional
        output path where the figure will be saved. The default is ./fig.
    title : string
        title of the plot
    density : boolean
        True to normalize the histogram, false otherwise.
    bins : int or string, optional
        bins for the function plt.hist(). The default is 'fd'.
    Returns
    -------
    plot
    """
    plt.figure(figsize=(10,7))
    plt.style.use('classic')
    #set uniform bins across all data points
    bins = np.histogram(np.hstack((ref,data)), bins = bins)[1]
    # reference
    hist_ref = plt.hist(ref, bins = bins, color=c_ref, edgecolor=e_ref,
                        density=density, label = 'Reference')
    x_err, err = err_bar(hist_ref, ref.shape[0])
    plt.errorbar(x_err, hist_ref[0], yerr = err, color=e_ref, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)
    # data
    hist_sig = plt.hist(data, bins = bins, color=c_sig, edgecolor=e_sig,
                        alpha=0.7, density=density, label='Data')
    x_err, err = err_bar(hist_sig, data.shape[0])
    plt.errorbar(x_err, hist_sig[0], yerr = err, color=e_sig, marker='o', ms=6, ls='', lw=1,
                 alpha=0.7)
    plt.ylim(bottom=0)
    # results data
    md_tref = round(np.median(ref), 2)
    md_tdata = round(np.median(data), 2)
    max_zemp = emp_zscore(ref,np.max(ref))
    zemp = emp_zscore(ref,np.median(data))
    if dof:
        z_chi2 = round(chi2_zscore(np.median(data),dof=dof),2)

    if dof:
        res = "md t_ref = {} \nmd t_data = {} \nmax Z_emp = {}  \nZ_emp = {} \nZ_chi2 = {}".format(
            md_tref,
            md_tdata,
            max_zemp,
            zemp,
            z_chi2
        )
    else:
        res = "md tref = {} \nmd tdata = {} \nmax Zemp = {} \nZemp = {}".format(
            md_tref,
            md_tdata,
            max_zemp,
            zemp
        )

    # plot chi2 and set xlim
    if dof:
        chi2_range = chi2.ppf(q=[0.00001,0.999], df=dof)
        #r_len = chi2_range[1] - chi2_range[0]
        x = np.arange(chi2_range[0], chi2_range[1], .05)
        chisq = chi2.pdf(x, df=dof)
        plt.plot(x, chisq, color='#d7191c', lw=2, label='$\chi^2(${}$)$'.format(dof))
        xlim = (min(chi2_range[0], min(ref)-1), max(chi2_range[1], max(data)+1))
        plt.xlim(xlim)
    else:
        xlim = (min(ref)-1, max(data)+1)
        plt.xlim(xlim)
    if title:
        plt.title(title, fontsize=20)

    plt.ylabel('P(t)', fontsize=20)
    plt.xlabel('t', fontsize=20)

    # Axes ticks
    ax = plt.gca()

    plt.legend(loc ="upper right", frameon=True, fontsize=18)

    ax.text(0.75, 0.55, res, color='black', fontsize=12,
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=.5'),transform = ax.transAxes)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path+"/refdata_{}.pdf".format(name), bbox_inches='tight')
    plt.close()
    
def run_toy(test_label, X_train, Y_train, weight, flk_config, seed,
            plot=False, verbose=False, savefig=False, output_path='', df=10, binsrange=None, yrange=None, xlabels=None):
    '''
    type of signal: "NP0", "NP1", "NP2", "NP3"
    output_path: directory (inside ./runs/) where to save results
    N_0: size of ref sample
    N0: expected num of bkg events
    NS: expected num of signal events
    flk_config: dictionary of logfalkon parameters
    toys: numpy array with seeds for toy generation
    plots_freq: how often to plot inputs with learned reconstructions
    df: degree of freedom of chi^2 for plots
    '''
    if not os.path.exists(output_path):
      os.makedirs(output_path, exist_ok=True)
    #save config file (temporary solution)
    with open(output_path+"/flk_config.txt","w") as f:
        f.write( str(flk_config) )
    dim = X_train.shape[1]
    # learn_t
    flk_config['seed']=seed # select different centers for different toys
    st_time = time.time()
    preds = trainer(X_train,Y_train,flk_config)
    t = compute_t(preds,Y_train,weight)
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---LRT = {}\n---Time = {} sec\n\t".format(seed,t,dt))
    with open(output_path+"t.txt", 'a') as f:
        f.write('{},{}\n'.format(seed,t))
    if plot:
        plot_reconstruction(data=X_train[Y_train.flatten()==1], weight_data=1,
                            ref=X_train[Y_train.flatten()==0], weight_ref=weight,
                            ref_preds=preds[Y_train.flatten()==0], #binsrange={'x': np.linspace(0, 50, 40),
                            yrange=yrange,binsrange=binsrange, xlabels=xlabels,
                            save=savefig, save_path=output_path+'/plots/', file_name=test_label+'.pdf'
                )
    return t, preds

def model_fitter(X,Y,flk_config):
    # trainer for logfalkon model                                                                                                                                                                                              
    Xtorch=torch.from_numpy(X)
    Ytorch=torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model#.predict(Xtorch).numpy()


def run_aggr_RD_1test(nbatches_ref, nbatches_dat, test_label, features_ref, features_dat,  weight, flk_config, seed, xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path='', return_predictions=False, return_predictions_all=False):
    '''
    density aggregation over data batches = arith-mean: F = log(mean(e^f))                                                                                      
    density aggregation over ref batches = arith-mean-inv: F = -log(mean(e^(-f)))                                                                               
    test only first batch of data
    weight: initial weight: N(R)/N_ref
    '''
    st_time = time.time()
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    run = 1
    features_all = np.concatenate((features_ref,
                                   features_dat[:int(len(s2)/nbatches_dat), :]), axis=0)
    target_all = np.concatenate((np.zeros(len(features_ref)),
                                 np.ones(int(len(features_dat)/nbatches_dat))), axis=0)
    F_br = np.zeros(len(features_ref)+int(len(features_dat)/nbatches_dat))
    for br in range(nbatches_ref):
        s1   = features_ref
        s1_b = s1[int(len(s1)/nbatches_ref)*br:int(len(s1)/nbatches_ref)*(br+1), :]
        if br==(nbatches_ref-1): s1_b = s1[int(len(s1)/nbatches_ref)*br:, :]
        F_bd = np.zeros(len(features_ref)+len(features_dat))
        for bd in range(nbatches_dat):
            s2   = features_dat
            s2_b = s2[int(len(s2)/nbatches_dat)*bd:int(len(s2)/nbatches_dat)*(bd+1), :]
            if bd==(nbatches_dat-1): s2_b = s2[int(len(s2)/nbatches_dat)*bd:, :]
            features  = np.concatenate((s2_b, s1_b), axis=0)
            label_R = np.zeros((len(s1_b),)).astype('f')
            label_D = np.ones((len(s2_b),)).astype('f')
            labels  = np.concatenate((label_D,label_R), axis=0)

            flk_config['seed']=seed*(run)
            #flk_config['weight']=weight*nbatches_ref*1./nbatches_dat                                                                                            

            model = model_fitter(features,labels,flk_config)

            preds = model.predict( torch.from_numpy(features_all) ).numpy()
            preds = preds.reshape((-1,))
            F_bd+=np.exp(preds)*1./nbatches_dat

            preds_local = model.predict(torch.from_numpy(features)).numpy()
            preds_local = preds_local.reshape((-1,))
            if bd==0:
                t_sum += compute_t(preds_local, labels,
                                   weight*nbatches_ref*1./nbatches_dat)*1./nbatches_ref
            run+=1
        F_bd = np.log(F_bd)
        F_br+=np.exp(-1*F_bd)*1./nbatches_ref
    F = -1*np.log(F_br)
    F_R = F[target_all==0]
    t_aggreg_D = compute_t(F, target_all, weight*1./nbatches_dat)
    t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---Time = {} sec\n\t".format(seed,dt))
    if plot:
        plot_reconstruction(data=features_all[target_all==1], weight_data=1,
                            ref=features_all[target_all==0], weight_ref=weight,
                            ref_preds=F[target_all==0].reshape((-1, 1)),
                            save=savefig, save_path=output_path+'/plots/',
                            file_name=test_label+'seed%i_all'%(seed)+'.pdf',
                            xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_aggreg_D, t_aggreg_R, t_sum
    
def run_aggr_RD(nbatches_ref, nbatches_dat, test_label, features_ref, features_dat,  weight, flk_config, seed, xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path='', return_predictions=False, return_predictions_all=False):
    '''
    density aggregation over data batches = arith-mean: F = log(mean(e^f)) 
    density aggregation over ref batches = arith-mean-inv: F = -log(mean(e^(-f)))
    '''
    st_time = time.time()
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    run = 1
    
    features_all = np.concatenate((features_ref, features_dat), axis=0)
    target_all = np.concatenate((np.zeros(len(features_ref)),
                                 np.ones(len(features_dat))), axis=0)
    F_br = np.zeros(len(features_ref)+len(features_dat))
    for br in range(nbatches_ref):
        s1   = features_ref
        s1_b = s1[int(len(s1)/nbatches_ref)*br:int(len(s1)/nbatches_ref)*(br+1), :]
        if br==(nbatches_ref-1): s1_b = s1[int(len(s1)/nbatches_ref)*br:, :]
        F_bd = np.zeros(len(features_ref)+len(features_dat))
        for bd in range(nbatches_dat):
            s2   = features_dat
            s2_b = s2[int(len(s2)/nbatches_dat)*bd:int(len(s2)/nbatches_dat)*(bd+1), :]
            if bd==(nbatches_dat-1): s2_b = s2[int(len(s2)/nbatches_dat)*bd:, :]
            features  = np.concatenate((s2_b, s1_b), axis=0)
            label_R = np.zeros((len(s1_b),)).astype('f')
            label_D = np.ones((len(s2_b),)).astype('f')
            labels  = np.concatenate((label_D,label_R), axis=0)

            flk_config['seed']=seed*(run)
            #flk_config['weight']=weight*nbatches_ref*1./nbatches_dat

            model = model_fitter(features,labels,flk_config)

            preds = model.predict( torch.from_numpy(features_all) ).numpy()
            preds = preds.reshape((-1,))
            F_bd+=np.exp(preds)*1./nbatches_dat
            
            preds_local = model.predict(torch.from_numpy(features)).numpy()
            preds_local = preds_local.reshape((-1,))
            t_sum += compute_t(preds_local, labels,
                               weight*nbatches_ref*1./nbatches_dat)*1./(nbatches_ref*nbatches_dat)
            run+=1
        F_bd = np.log(F_bd)
        F_br+=np.exp(-1*F_bd)*1./nbatches_ref
    F = -1*np.log(F_br)
    F_R = F[target_all==0]
    t_aggreg_D = compute_t(F, target_all, weight)
    t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---Time = {} sec\n\t".format(seed,dt))
    if plot:
        plot_reconstruction(data=features_all[target_all==1], weight_data=1,
                            ref=features_all[target_all==0], weight_ref=weight,
                            ref_preds=F[target_all==0].reshape((-1, 1)),
                            save=savefig, save_path=output_path+'/plots/',
                            file_name=test_label+'seed%i_all'%(seed)+'.pdf',
                            xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_aggreg_D, t_aggreg_R, t_sum
    
def run_aggr(n_aggreg, test_label, X_train, Y_train, weight, flk_config, seed, xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path='', return_predictions=False, return_predictions_all=False):
    '''                                                                                                                                                                                      
    type of signal: "NP0", "NP1", "NP2", "NP3"                                                                                                                                               
    output_path: directory (inside ./runs/) where to save results                                                                                                                       
    N_0: size of ref sample                                                                                                                                       
    N0: expected num of bkg events                                                                                                                                                  
    NS: expected num of signal events                                                                                                                                                     
    flk_config: dictionary of logfalkon parameters                                                                                                                     
    toys: numpy array with seeds for toy generation                                                                                                                     
    plots_freq: how often to plot inputs with learned reconstructions                                                                                                                  
    df: degree of freedom of chi^2 for plots                                                                                                                        
    density aggregation = arith-mean: F = log(mean(e^f))
    '''
    # learn_t
    preds_dict = {}
    st_time = time.time()
    for n in range(n_aggreg):
        preds_dict[str(n)] = {}
        flk_config['seed']=seed*(n+1) # select different centers for different toys                                                                                                                                           
        #st_time = time.time()
        dim = X_train[n].shape[1]
        model = model_fitter(X_train[n],Y_train[n],flk_config)
        for m in range(n_aggreg):
            preds = model.predict( torch.from_numpy(X_train[m]) ).numpy()
            preds_dict[str(n)][str(m)]= preds.reshape((-1, 1))
    FR_dict = {}
    FD_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    for m in range(n_aggreg):
        F = np.zeros_like(preds_dict['0'][str(m)])
        for n in range(n_aggreg):
            F+=np.exp(preds_dict[str(n)][str(m)])*1./n_aggreg
        F = np.log(F)
        F_R = F[Y_train[m]==0]
        F_D = F[Y_train[m]==1]
        FR_dict[str(m)] = F_R.reshape((-1, 1))
        FD_dict[str(m)] = F_D.reshape((-1, 1))
        t_aggreg_D += compute_t(F, Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
        t_sum += compute_t(preds_dict[str(m)][str(m)], Y_train[m], weight)*1./n_aggreg
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---Time = {} sec\n\t".format(seed,dt))
        
    if plot:
        for m in range(n_aggreg):
            #print(FR_dict[str(m)].shape, X_train[m][Y_train[m].flatten()==1].shape, preds_dict[str(0)][str(m)][Y_train[m].flatten()==0].shape)
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_aggr%i'%(seed, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
            plot_reconstruction(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=preds_dict[str(m)][str(m)][Y_train[m].flatten()==0], 
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_single%i'%(seed, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
        plot_reconstruction_ensemble(data=np.concatenate(([X_train[m][Y_train[m].flatten()==1] for m in range(n_aggreg)]), axis=0),
                                     weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight*n_aggreg,
                                     ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                     save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_all-aggr%i'%(seed, m)+'.pdf',
                                     xlabels=xlabels_list, yrange=yrange_dict
            )
    if return_predictions_all:
        return t_sum, t_aggreg_D, t_aggreg_R, FR_dict, FD_dict
    elif return_predictions:
        return t_sum, t_aggreg_D, t_aggreg_R, FR_dict
    else:
        return t_sum, t_aggreg_D, t_aggreg_R

def run_aggr_geom(n_aggreg, test_label, X_train, Y_train, weight, flk_config, seed, xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    '''                                                                                                                                                                     
    type of signal: "NP0", "NP1", "NP2", "NP3"                                                                                                                   
    output_path: directory (inside ./runs/) where to save results                                                                                                           
                                                                                                                                                                            
    N_0: size of ref sample                                                                                                                                                 
    N0: expected num of bkg events                                                                                                                                          
                                                                                                                                                                            
    NS: expected num of signal events                                                                                                                                       
                                                                                                                                                                            
    flk_config: dictionary of logfalkon parameters                                                                                                                          
    toys: numpy array with seeds for toy generation                                                                                                                         
    plots_freq: how often to plot inputs with learned reconstructions                                                                                                                                                                                                                                                                                  
    df: degree of freedom of chi^2 for plots                                                                                                                            
                                                                                                                                                               
    density aggregation = geom-mean: F = mean(f)
    '''
    # learn_t                                                                                                                                                               
    preds_dict = {}
    for n in range(n_aggreg):
        preds_dict[str(n)] = {}
        flk_config['seed']=seed*(n+1) # select different centers for different toys                                                                                
        
        st_time = time.time()
        dim = X_train[n].shape[1]
        model = model_fitter(X_train[n],Y_train[n],flk_config)
        for m in range(n_aggreg):
            preds = model.predict( torch.from_numpy(X_train[m]) ).numpy()
            preds_dict[str(n)][str(m)]= preds.reshape((-1, 1))
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    for m in range(n_aggreg):
        F = np.zeros_like(preds_dict['0'][str(m)])
        for n in range(n_aggreg):
            F+=preds_dict[str(n)][str(m)]*1./n_aggreg
        F_R = F[Y_train[m]==0]
        FR_dict[str(m)] = F_R.reshape((-1, 1))
        t_aggreg_D += compute_t(F, Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
        t_sum += compute_t(preds_dict[str(m)][str(m)], Y_train[m], weight)*1./n_aggreg
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---Time = {} sec\n\t".format(seed,dt))

    if plot:
        for m in range(n_aggreg):
            #print(FR_dict[str(m)].shape, X_train[m][Y_train[m].flatten()==1].shape, preds_dict[str(0)][str(m)][Y_train[m].flatten()==0].shape)                              
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_aggr%i'%(seed, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
            plot_reconstruction(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=preds_dict[str(m)][str(m)][Y_train[m].flatten()==0],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_single%i'%(seed, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
        plot_reconstruction_ensemble(data=np.concatenate(([X_train[m][Y_train[m].flatten()==1] for m in range(n_aggreg)]), axis=0),
                                     weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight*n_aggreg,
                                     ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                     save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_all-aggr%i'%(seed, m)+'.pdf',
                                     xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R

def run_aggr3(toy, n_aggreg, test_label, X_train, Y_train,  weight, flk_seeds, flk_sigmas, flk_M, flk_lam, flk_dfs,
              xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    preds_dict = {}
    for n in range(n_aggreg):
        preds_dict[str(n)] = {}
        pval_tmp = np.array([])
        model_tmp = np.array([])
        for i in range(len(flk_sigmas)):
            #preds_dict[str(n)][str(flk_sigma)]={}
            flk_config = get_logflk_config(flk_M,flk_sigmas[i],[flk_lam],weight=weight,iter=[1000000],seed=flk_seeds[i],cpu=False) 
            st_time = time.time()
            dim = X_train[n].shape[1]
            model = model_fitter(X_train[n],Y_train[n],flk_config)
            preds = model.predict( torch.from_numpy(X_train[n]) ).numpy()
            t = compute_t(preds, Y_train[n], weight)
            pval_tmp = np.append(pval_tmp, chi2.sf(t, flk_dfs[i]))
            model_tmp = np.append(model_tmp, model)
        # select sigma with min p-value
        j_minp = np.argmin(pval_tmp)
        model_min = model_tmp[j_minp]
        for m in range(n_aggreg):
            preds = model_min.predict( torch.from_numpy(X_train[m]) ).numpy()
            preds_dict[str(n)][str(m)]= preds.reshape((-1, 1))
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    for m in range(n_aggreg):
        F = np.zeros_like(preds_dict['0'][str(m)])
        for n in range(n_aggreg):
            F+=np.exp(preds_dict[str(n)][str(m)])*1./n_aggreg
        F = np.log(F)
        F_R = F[Y_train[m]==0]
        FR_dict[str(m)] = F_R.reshape((-1, 1))
        t_aggreg_D += compute_t(F, Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
        t_sum += compute_t(preds_dict[str(m)][str(m)], Y_train[m], weight)*1./n_aggreg
    dt = round(time.time()-st_time,2)
    if verbose:
        print("---Time = {} sec\n\t".format(dt))
    if plot:
        for m in range(n_aggreg):
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'toy%i-aggr%i'%(toy, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R

def run_aggr2(toy, n_aggreg, test_label, X_train, Y_train,  weight, flk_seeds, flk_sigmas, flk_M, flk_lam, flk_dfs,
              xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    preds_dict = {}
    for i in range(len(flk_sigmas)):
        preds_dict[str(flk_sigmas[i])] = {}
        #pval_tmp = np.array([])
        #model_tmp = np.array([])
        #preds_tmp = np.array([])
        for n in range(n_aggreg):
            preds_dict[str(flk_sigmas[i])][str(n)] ={}
            flk_config = get_logflk_config(flk_M,flk_sigmas[i],[flk_lam],weight=weight,iter=[1000000],seed=flk_seeds[i]*n,cpu=False)
            st_time = time.time()
            dim = X_train[n].shape[1]
            model = model_fitter(X_train[n],Y_train[n],flk_config)
            #preds = model.predict( torch.from_numpy(X_train[n]) ).numpy()
            #t = compute_t(preds, Y_train[n], weight)
            for m in range(n_aggreg):
                preds = model.predict( torch.from_numpy(X_train[m]) ).numpy()
                preds_dict[str(flk_sigmas[i])][str(n)][str(m)]= preds.reshape((-1, 1))
        #pval_tmp = np.append(pval_tmp, chi2.sf(t, flk_dfs[i]))
        #model_tmp = np.append(model_tmp, model)
        # select sigma with min p-value
        #j_minp = np.argmin(pval_tmp)
        #model_min = model_tmp[j_minp]
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    F_dict= {}
    for m in range(n_aggreg):
        F_dict[str(m)] = np.zeros_like(preds_dict[str(flk_sigmas[0])]['0'][str(m)])
        for n in range(n_aggreg):
            for i in range(len(flk_sigmas)):
                F_dict[str(m)]+=np.exp(preds_dict[str(flk_sigmas[i])][str(n)][str(m)])*1./(n_aggreg*len(flk_sigmas))
        F_dict[str(m)] = np.log(F_dict[str(m)])
    for m in range(n_aggreg):
        FR_dict[str(m)] = F_dict[str(m)][Y_train[m]==0]
        FR_dict[str(m)] = FR_dict[str(m)].reshape((-1, 1))
        t_aggreg_D += compute_t(F_dict[str(m)], Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(FR_dict[str(m)])*FR_dict[str(m)] +1 -np.exp(FR_dict[str(m)]))
        for i in range(len(flk_sigmas)):
            t_sum += compute_t(preds_dict[str(flk_sigmas[i])][str(m)][str(m)], Y_train[m], weight)*1./(n_aggreg*len(flk_sigmas))
    dt = round(time.time()-st_time,2)
    if verbose:
        print("---Time = {} sec\n\t".format(dt))
    if plot:
        for m in range(n_aggreg):
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(flk_sigmas[i])][str(m)][str(m)][Y_train[m].flatten()==0] for i in range(len(flk_sigmas))], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'toy%i-aggr%i'%(toy, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R


def run_aggr4(toy, n_aggreg, test_label, X_train, Y_train,  weight, flk_seeds, flk_sigmas, flk_M, flk_lam, flk_dfs,
              xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    preds_dict = {}
    for i in range(len(flk_sigmas)):
        preds_dict[str(flk_sigmas[i])] = {}
        #pval_tmp = np.array([])                                                                                                                                                                                               
        #model_tmp = np.array([])                                                                                                                                                                                              
        #preds_tmp = np.array([])                                                                                                                                                                                              
        for n in range(n_aggreg):
            preds_dict[str(flk_sigmas[i])][str(n)] ={}
            flk_config = get_logflk_config(flk_M,flk_sigmas[i],[flk_lam],weight=weight,iter=[1000000],seed=flk_seeds[i]*n,cpu=False)
            st_time = time.time()
            dim = X_train[n].shape[1]
            model = model_fitter(X_train[n],Y_train[n],flk_config)
            #preds = model.predict( torch.from_numpy(X_train[n]) ).numpy()                                                                                                                                                     
            #t = compute_t(preds, Y_train[n], weight)                                                                                                                                                                          
            for m in range(1):
                preds = model.predict( torch.from_numpy(X_train[m]) ).numpy()
                preds_dict[str(flk_sigmas[i])][str(n)][str(m)]= preds.reshape((-1, 1))
        #pval_tmp = np.append(pval_tmp, chi2.sf(t, flk_dfs[i]))                                                                                                                                                                
        #model_tmp = np.append(model_tmp, model)                                                                                                                                                                               
        # select sigma with min p-value                                                                                                                                                                                        
        #j_minp = np.argmin(pval_tmp)                                                                                                                                                                                          
        #model_min = model_tmp[j_minp]                                                                                                                                                                                         
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    F_dict= {}
    for m in range(1):
        F_dict[str(m)] = np.zeros_like(preds_dict[str(flk_sigmas[0])]['0'][str(m)])
        for n in range(n_aggreg):
            for i in range(len(flk_sigmas)):
                F_dict[str(m)]+=np.exp(preds_dict[str(flk_sigmas[i])][str(n)][str(m)])*1./(n_aggreg*len(flk_sigmas))
        F_dict[str(m)] = np.log(F_dict[str(m)])
    for m in range(1):
        FR_dict[str(m)] = F_dict[str(m)][Y_train[m]==0]
        FR_dict[str(m)] = FR_dict[str(m)].reshape((-1, 1))
        t_aggreg_D += compute_t(F_dict[str(m)], Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(FR_dict[str(m)])*FR_dict[str(m)] +1 -np.exp(FR_dict[str(m)]))
        for i in range(len(flk_sigmas)):
            t_sum += compute_t(preds_dict[str(flk_sigmas[i])][str(m)][str(m)], Y_train[m], weight)*1./(n_aggreg*len(flk_sigmas))
    dt = round(time.time()-st_time,2)
    if verbose:
        print("---Time = {} sec\n\t".format(dt))
    if plot:
        for m in range(1):
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(flk_sigmas[i])][str(m)][str(m)][Y_train[m].flatten()==0] for i in range(len(flk_sigmas))], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'toy%i-aggr%i'%(toy, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R

def run_aggr5(toy, n_aggreg, test_label, X_train, Y_train,  weight, flk_seeds, flk_sigmas, flk_M, flk_lam, flk_dfs,
              xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    '''
    MINp-sigma, AVGf-b, (test 1 batch)
    '''
    preds_dict = {}
    for n in range(n_aggreg):
        preds_dict[str(n)] = {}
        pval_tmp = np.array([])
        model_tmp = np.array([])
        for i in range(len(flk_sigmas)):
            #preds_dict[str(n)][str(flk_sigma)]={}                                                                                                                                                                                                                 
            flk_config = get_logflk_config(flk_M,flk_sigmas[i],[flk_lam],weight=weight,iter=[1000000],seed=flk_seeds[i],cpu=False)
            st_time = time.time()
            dim = X_train[n].shape[1]
            model = model_fitter(X_train[n],Y_train[n],flk_config)
            preds = model.predict( torch.from_numpy(X_train[n]) ).numpy()
            t = compute_t(preds, Y_train[n], weight)
            pval_tmp = np.append(pval_tmp, chi2.sf(t, flk_dfs[i]))
            model_tmp = np.append(model_tmp, model)
        # select sigma with min p-value                                                                                                                                                                                                                            
        j_minp = np.argmin(pval_tmp)
        model_min = model_tmp[j_minp]
        for m in range(1):
            preds = model_min.predict( torch.from_numpy(X_train[m]) ).numpy()
            preds_dict[str(n)][str(m)]= preds.reshape((-1, 1))
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    for m in range(1):
        F = np.zeros_like(preds_dict['0'][str(m)])
        for n in range(n_aggreg):
            F+=np.exp(preds_dict[str(n)][str(m)])*1./n_aggreg
        F = np.log(F)
        F_R = F[Y_train[m]==0]
        FR_dict[str(m)] = F_R.reshape((-1, 1))
        t_aggreg_D += compute_t(F, Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
        t_sum += compute_t(preds_dict[str(m)][str(m)], Y_train[m], weight)*1./n_aggreg
    dt = round(time.time()-st_time,2)
    if verbose:
        print("---Time = {} sec\n\t".format(dt))
    if plot:
        for m in range(1):
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'toy%i-aggr%i'%(toy, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R

def run_aggr6(n_aggreg, test_label, X_train, Y_train,  weight, seed, flk_config,
              xlabels_list=[], yrange_dict=None, plot=False, verbose=False, savefig=False, output_path=''):
    '''
    AVGf-b, MINp-sigma, (test 1 batch)
    '''
    preds_dict = {}
    st_time = time.time()
    for n in range(n_aggreg):
        preds_dict[str(n)] = {}
        flk_config['seed']=seed*(n+1) # select different centers for different toys                                                                                   
        dim = X_train[n].shape[1]
        model = model_fitter(X_train[n],Y_train[n],flk_config)
        for m in range(1):
            preds = model.predict( torch.from_numpy(X_train[m]) ).numpy()
            preds_dict[str(n)][str(m)]= preds.reshape((-1, 1))
    FR_dict = {}
    t_aggreg_D = 0
    t_aggreg_R = 0
    t_sum = 0
    for m in range(1):
        F = np.zeros_like(preds_dict['0'][str(m)])
        for n in range(n_aggreg):
            F+=np.exp(preds_dict[str(n)][str(m)])*1./n_aggreg
        F = np.log(F)
        F_R = F[Y_train[m]==0]
        FR_dict[str(m)] = F_R.reshape((-1, 1))
        t_aggreg_D += compute_t(F, Y_train[m], weight)
        t_aggreg_R = 2*np.sum(np.exp(F_R)*F_R +1 -np.exp(F_R))
        t_sum += compute_t(preds_dict[str(m)][str(m)], Y_train[m], weight)*1./n_aggreg
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---Time = {} sec\n\t".format(seed,dt))

    if plot:
        for m in range(1):
            #print(FR_dict[str(m)].shape, X_train[m][Y_train[m].flatten()==1].shape, preds_dict[str(0)][str(m)][Y_train[m].flatten()==0].shape)                                                                                                                                               
            plot_reconstruction_ensemble(data=X_train[m][Y_train[m].flatten()==1], weight_data=1, ref=X_train[m][Y_train[m].flatten()==0], weight_ref=weight,
                                         ref_preds=[preds_dict[str(n)][str(m)][Y_train[m].flatten()==0] for n in range(n_aggreg)], ref_preds_aggreg=FR_dict[str(m)],
                                         save=savefig, save_path=output_path+'/plots/', file_name=test_label+'seed%i_aggr%i'%(seed, m)+'.pdf',
                                         xlabels=xlabels_list, yrange=yrange_dict
            )
    return t_sum, t_aggreg_D, t_aggreg_R

def plot_reconstruction_ensemble(data, weight_data, ref, weight_ref, ref_preds_aggreg, ref_preds=[],t_obs=None, df=None, title='', xlabels=[], yrange=None, binsrange=None,
                        save=False, save_path='', file_name=''):
    '''                                                                                                                                                                                                                      
    Reconstruction of the data distribution learnt by the model.                                                                                                                                                                                                                                                                                                                                                                     
    df:              (int) chi2 degrees of freedom                                                                                                                                                                            
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)                                                                                                                                 
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)                                                                                                                                   
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)                                                                                                                            
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample                                                                                                                                             
    tau_OBS:         (float) value of the tau term after training                                                                                                                                                                 output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training                                                                                                          
    feature_labels:  (list of string) list of names of the training variables                                                                                                                                                 
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)                                                                                                              
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)                                                                                                                    
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)                                                                        
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)                                                                                                        
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training (if not given, only tau reconstruction is plotted)                                                       
    '''
    # used to regularize empty reference bins                                                                                                                                                        
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data
    
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        bins = np.linspace(np.min(ref[:, i]),np.max(ref[:, i]),50)
        if not binsrange==None:
            if len(binsrange[xlabels[i]]):
                bins=binsrange[xlabels[i]]
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.13, 0.43, 0.8, 0.5])
        hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = plt.hist(ref[:, i], weights=np.exp(ref_preds_aggreg[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)
        hN_ensemble = []
        if len(ref_preds):
            for j in range(len(ref_preds)):
                hN_ens = plt.hist(ref[:, i], weights=np.exp(ref_preds[j][:, 0])*weight_ref, histtype='step', bins=bins, lw=0)
                hN_ensemble.append(hN_ens)
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='RECO', color='#33a02c', lw=1, s=30, zorder=4)

        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
        font = font_manager.FontProperties(family='serif', size=18)
        title_leg=''
        if not t_obs==None:
            title_leg += 't='+str(np.around(t_obs, 2))
        if not df==None:
            Zscore=norm.ppf(chi2.cdf(t_obs, df))
            title_leg += ', Z-score='+str(np.around(Zscore, 2))
        if not title_leg=='':
            l.set_title(title=title_leg, prop=font)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        if title!='':
            plt.title(title, fontsize=22, fontname='serif')
        if len(xlabels):
            if '_pt' in xlabels[i] or 'm_' in xlabels[i]:
                plt.xlim(bins[0]+0.1, bins[-1])
                plt.xscale('log')
        ax2 = fig.add_axes([0.13, 0.1, 0.8, 0.3])
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black')
        if len(hN_ensemble):
            for j in range(len(hN_ensemble)):
                y=hN_ensemble[j][0]/(hR[0]+eps)
                plt.plot(x[y>0], y[y>0], color='#b2df8a', alpha=0.5, lw=2)
        y = hN[0]/(hR[0]+eps)
        plt.plot(x[y>0], y[y>0], label ='AGGR RECO/REF', color='#33a02c', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(loc='best', prop=font)
        if len(xlabels)>0:
            plt.xlabel(xlabels[i], fontsize=22, fontname='serif')
        else:
            plt.xlabel('x', fontsize=22, fontname='serif')
            
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        if len(xlabels):
            if '_pt' in xlabels[i] or 'm_' in xlabels[i]:
                plt.xlim(bins[0]+0.1, bins[-1])
                plt.xscale('log')
        if not yrange==None and len(xlabels)>0:
            plt.ylim(yrange[xlabels[i]][0], yrange[xlabels[i]][1])                                                                                                                                                                                                          
        plt.grid()
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path+file_name.replace('.pdf', '_%i.pdf'%(i)))
        plt.show()
        plt.close()
    return
