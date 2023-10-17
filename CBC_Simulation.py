#!/usr/bin/env python

import numpy as np
import pandas as pd
import scipy.signal as sig
from numpy.random import default_rng
import threading
import time
import json
from itertools import combinations, chain
from multiprocessing import Lock,RLock,Array
from tqdm import tqdm
import os
import multiprocessing
import corner
import scipy.optimize as optimize
import argparse
import scipy.stats as st
import GWFish.modules as gw
import GWFish.modules.constants as cst
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
rng = default_rng()
lock=Lock()
rlock=RLock()
cond=threading.Condition()
flag_ins_GR=0
flag_MR_GR=0


def pre_ins():
    print ("flag ins GR")
    return flag_ins_GR
def pre_MR():
    print ("flag MR GR")
    return flag_MR_GR


def get_ci(samples, ci=0.9):
    # compute 90% CL
    #samples=df.values
    med = np.percentile(samples, 50)
    lo =  np.percentile(samples, 50*(1-ci))
    hi =  np.percentile(samples, 100-50*(1-ci))
    return med, lo, hi
def powerset(length):
    it = chain.from_iterable((combinations(range(length), r)) for r in range(length+1))
    return list(it)[1:]

def plot_rv_set(pmean,cov,pdfs,PDF_For_L):
    rv=st.multivariate_normal(pmean,cov)
    x, y = np.mgrid[57:65.5:0.01, 0.67:0.73:0.0001]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    PDF=rv.pdf(pos)
    #print("PDF",np.shape(PDF))
    #PDF_For_L.append(PDF)
    max_prob_index_M = np.argmax(rv.pdf(pos),axis=0)

    PDF_prob_y=[]
    for i in range(len(max_prob_index_M)):
        PDF_prob_y.append(PDF[max_prob_index_M[i]][i])
    PDF_prob_y=np.array(PDF_prob_y)
    #print("PDF_prob_y",PDF_prob_y)

    max_prob_index_chi=np.argmax(PDF_prob_y)
    pdfs_per_max={'prob':PDF[max_prob_index_M[max_prob_index_chi]][max_prob_index_chi],
                  'M_chif':(57+max_prob_index_M[max_prob_index_chi]*0.01,0.67+max_prob_index_chi*0.0001)}
    pdfs.append(pdfs_per_max)


def plot_Mf_chif(pmean,cov,flag,paraGR):
    print("current thread{},pmean{}".format(multiprocessing.current_process().name,pmean))
    print("current thread{},cov{}".format(multiprocessing.current_process().name,cov))
    #with lock:
    print("current plor flag",flag)
    rv = st.multivariate_normal(pmean, cov)
    x, y = np.mgrid[57:65.5:0.01, 0.67:0.73:0.0001]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    max_prob = np.max(rv.pdf(pos))
    color=['green','purple','red','cyan']
    lstyles=['dashed','solid']
    if 'ins' in flag:
        lstyle=lstyles[0]
    elif 'MR' in flag:
        lstyle=lstyles[1]
    CS = plt.contour(x, y, rv.pdf(pos), [max_prob * 0.1, max_prob],colors=color[int(flag[-1])],linestyles=lstyle)
    #plt.clabel(CS, inline=1, fontsize=10)
    # CS.collections[0].set_label('{}'.format(flag))
    # plt.contourf(x_m, y_chi, rv_1.pdf(pos))
    plt.scatter(paraGR[0], paraGR[1], marker='+')
    plt.annotate('GR', xy=(paraGR[0], paraGR[1]))

    #plt.scatter(pmean[0],pmean[1],marker='+')
    #plt.annotate('epsilon_{}'.format(flag[-1]),xy=(pmean[0],pmean[1]))
    #plt.colorbar()
    #plt.title('{} Mf chif'.format(flag))
    plt.xlabel('Mf')
    plt.ylabel('Chif')
    #plt.savefig('{}_Mf_chif__22_plot.png'.format(flag))
def createProcessforEpsilon(flag,epsilon):
    #print("create thread epsilon")
    flags=[]
    for i in range(len(epsilon)):
        flags.append(flag+'_{}'.format(i))
    #print("flag",flag)
    createSuccess=1
    t0 = multiprocessing.Process(target=main,args=(flags[0],createSuccess),name=flags[0])
    t1 = multiprocessing.Process(target=main, args=(flags[1], createSuccess),name=flags[1])
    t2 = multiprocessing.Process(target=main, args=(flags[2], createSuccess),name=flags[2])
    t3 = multiprocessing.Process(target=main, args=(flags[3], createSuccess),name=flags[3])
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
def wait_for_GR(flag):
    # GR进程未完成，则一直等待
    if flag[:-2]=='ins_Modify':
        while(1):
            with open("ins_GR_Complete.txt","r") as f:
                if len(f.read())!=0:
                    complete_flag = np.loadtxt("ins_GR_Complete.txt", dtype=int)
                    print("complete flag in ins:",complete_flag)
                    if complete_flag==None or complete_flag==0:
                        continue
                    if complete_flag==1:
                        break


    elif flag[:-2] == 'MR_Modify':
        while (1):
            with open("ins_GR_Complete.txt", "r") as f:
                if len(f.read()) != 0:
                    complete_flag_MR = np.loadtxt("MR_GR_Complete.txt", dtype=int)
                    print("complete_flag", complete_flag_MR)
                    if complete_flag_MR == None or complete_flag_MR == 0:
                        continue
                    if complete_flag_MR == 1:
                        break
    print("release:",flag)

def GR_process(flag):
    #GR进程写入保存完成，将flag置为1
    if flag=='ins_Modify_0':
        flag_ins_GR=[1]
        np.savetxt("ins_GR_Complete.txt",flag_ins_GR,fmt='%d')
        print("Ins GR save")
    elif flag=='MR_Modify_0':
        flag_MR_GR=[1]
        np.savetxt("MR_GR_Complete.txt", flag_MR_GR,fmt='%d')
        print ("MR GR save")

def plot_corner(mean,cov,flag):
    CORNER_KWARGS = dict(
        bins=50,  # number of bins for histograms
        smooth=0.99,  # smooths out contours.
        plot_datapoints=True,  # choose if you want datapoints
        label_kwargs=dict(fontsize=12),  # font size for labels
        show_titles=True,  # choose if you want titles on top of densities.
        title_kwargs=dict(fontsize=12),  # font size for title
        plot_density=False,
        title_quantiles=[0.16, 0.5, 0.84],  # add quantiles to plot densities for 1d hist
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),  # 1, 2 and 3 sigma contours for 2d plots
        fill_contours=True,  # decide if you want to fill the contours
        max_n_ticks=2,  # set a limit to ticks in the x-y axes.
        title_fmt=".3f"
    )
    corner_lbs = [r'$lnAGR$', '$phase$','$eta$', '$chi_s$ ', '$Mf$ $[M_{\odot}]$', '$chif$']
    """
    mean_values = [parameters['mass_1'].iloc[0], parameters['mass_2'].iloc[0],
                   parameters['luminosity_distance'].iloc[0],
                   parameters['theta_jn'].iloc[0], parameters['dec'].iloc[0],
                   parameters['ra'].iloc[0], parameters['psi'].iloc[0],
                   parameters['phase'].iloc[0], parameters['geocent_time'].iloc[0],
                   parameters['a_1'].iloc[0], parameters['a_2'].iloc[0],
                   parameters['lambda_1'].iloc[0], parameters['lambda_2'].iloc[0]]
    """
    # Sample from a multi-variate gaussian with the given covariance matrix and injected mean values
    samples = np.random.multivariate_normal(mean, cov, int(1e5))
    fig = corner.corner(samples, labels=corner_lbs, truths=mean, truth_color='red',
                        **CORNER_KWARGS)
    plt.savefig('corner_plt_{}.png'.format(flag))
    #plt.show()
def PlanckTaperWindow(freqVec,flag):
    #Planck窗口，在波形做反傅里叶变换之前，需要加窗函数，一般信号处理中使用的窗函数为汉明窗这些。
    #print("{}len freq {}".format(flag,len(freqVec)))
    f10 = freqVec[0][0]
    f1 = []
    f1_temp = f10
    f1.append(f1_temp)
    while f1_temp <= freqVec[-1]:
        f1_temp *= (5. / 4.)
        f1.append(f1_temp)
    f1 = np.array(f1)
    f2 = f1 / 0.8
    f2[-1] = freqVec[-1]
    PlanckTaperWindowVal = []
    for i in range(0, len(f2)):
        for j in range(0, len(freqVec)):
            if freqVec[j] <= f2[i] and freqVec[j] >= f1[i]:
                if freqVec[j]-f2[i]<=0.0001:
                    freqtemp = (freqVec[j] + freqVec[j - 1]) / 2
                    PlanckTaperWindowVal.append(1. / (1 + np.exp((f2[i] - f1[i]) / (freqtemp - f1[i]) + (f2[i] - f1[i]) / (freqtemp - f2[i]))))
                elif freqVec[j]-f1[i]<=0.0001:
                    freqtemp = (freqVec[j] + freqVec[j + 1]) / 2
                    PlanckTaperWindowVal.append(1. / (1 + np.exp((f2[i] - f1[i]) / (freqtemp - f1[i]) + (f2[i] - f1[i]) / (freqtemp - f2[i]))))
                else:
                    WindowsVal = 1. / (1 + np.exp((f2[i] - f1[i]) / (freqVec[j] - f1[i]) + (f2[i] - f1[i]) / (freqVec[j] - f2[i])))
                    PlanckTaperWindowVal.append(WindowsVal)
    PlanckTaperWindowVal = np.array(PlanckTaperWindowVal)
    #print("{}len PlanckTaperWindowVal {}".format(flag, len(PlanckTaperWindowVal)))
    return PlanckTaperWindowVal
def fun_cos(t,A,omega,phi,C):
    return A*np.cos(omega*t + phi) + C

def main(flag,alread_create_epsilon=0):
    # example to run with command-line arguments:
    # python CBC_Simulation.py --pop_file=CBC_pop.hdf5 --detectors ET CE2 --networks [[0,1],[0],[1]]
    print("start progra ",flag)
    epsilon=[0,0.04,0.05,0.08]
    if alread_create_epsilon==0:
        createProcessforEpsilon(flag,epsilon)
    if flag=='ins_Modify' or flag=='MR_Modify':
        return
    index=int(flag[-1])

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--pop_file', type=str, default='./injections/150914_like_population_{}.hdf5'.format(flag),
        help='Population to run the analysis on.'
             'Runs on BBH_1e5.hdf5 if no argument given.')
    parser.add_argument(
        '--pop_id', type=str, default='BBH',
        help='Short population identifier for file names. Uses BBH if no argument given.')
    parser.add_argument(
        '--detectors', type=str, default=['CE1'], nargs='+',
        help='Detectors to analyze. Uses ET as default if no argument given.')
    parser.add_argument(
        '--networks', default='all',
        help='''Network IDs: list of lists of detector IDs. 
Uses [[0]] (only the first detector) as default if no argument given.
Use "all" to get all possible combinations of the detectors given.''')
    parser.add_argument(
        '--config', type=str, default='GWFish/detectors.yaml',
        help='Configuration file where the detector specifications are stored. Uses GWFish/detectors.yaml as default if no argument given.')

    #清空flag .txt文件
    c_flag=[0]
    np.savetxt("ins_GR_Complete.txt",c_flag,fmt='%d')
    np.savetxt("MR_GR_Complete.txt", c_flag,fmt='%d')
    args = parser.parse_args()
    ConfigDet = args.config

    threshold_SNR = np.array([0., 9.])  # [min. individual SNR to be included in PE, min. network SNR for detection]
    calculate_errors = True   # whether to calculate Fisher-matrix based PE errors
    duty_cycle = False  # whether to consider the duty cycle of detectors

    #fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'mass_1', 'mass_2', 'geocent_time', 'phase']
    #fisher_parameters = ['ra', 'dec', 'psi', 'theta_jn', 'luminosity_distance', 'Mf', 'af']
    fisher_parameters = ['lnAGR','phase','geocent_time','eta','chi_s','epsilon','Mf', 'af']
    #fisher_parameters=['eta','chi_s','Mf', 'af']
    #fisher_parameters = ['Mf', 'af']
    #fisher_parameters = ['luminosity_distance','ra','dec']

    pop_file = args.pop_file
    population = args.pop_id

    detectors_ids = args.detectors
    if args.networks == 'all':
        networks_ids = powerset(len(detectors_ids))
    else:
        networks_ids = json.loads(args.networks)

    parameters = pd.read_hdf(pop_file)
    network = gw.detection.Network(detectors_ids, detection_SNR=threshold_SNR, parameters=parameters,
                                   fisher_parameters=fisher_parameters, config=ConfigDet)

    # lisaGWresponse(network.detectors[0], frequencyvector)
    # exit()

    # horizon(network, parameters.iloc[0], frequencyvector, threshold_SNR, 1./df, fmax)
    # exit()

    #waveform_model = 'gwfish_TaylorF2'
    waveform_model = 'gwfish_IMRPhenomD'
    #waveform_model = 'lalsim_TaylorF2'
    #waveform_model = 'lalsim_IMRPhenomD'
    #waveform_model = 'lalsim_IMRPhenomXPHM'
    #########################################################################################################
    np.random.seed(0)

    once_flag=0
    nd=len(fisher_parameters)
    amp_data_all=[]
    phase_data_all=[]
    print('Processing CBC population:',threading.current_thread().name)
    for k in tqdm(np.arange(len(parameters))):
        parameter_values = parameters.iloc[k]
        #print("para values",parameter_values)
        networkSNR_sq = 0
        for d in np.arange(len(network.detectors)):
            once_flag+=1
            wave, t_of_f = gw.waveforms.hphc_amplitudes(waveform_model, parameter_values,
                                                        network.detectors[d].frequencyvector,once_flag,flag=flag)
                                                        #plot=network.detectors[d].plotrange)
            Fisco=np.loadtxt("Fisco_{}.csv".format(flag))
            ret=np.where(network.detectors[d].frequencyvector>=Fisco)
            plot_f2=ret[0][0]

            once_flag=0
            signal = gw.detection.projection(parameter_values, network.detectors[d], wave, t_of_f)

            #print("{} signals {}".format(threading.current_thread().name, signal))
            SNRs = gw.detection.SNR(network.detectors[d], signal, duty_cycle=duty_cycle)
            networkSNR_sq += np.sum(SNRs ** 2)
            network.detectors[d].SNR[k] = np.sqrt(np.sum(SNRs ** 2))

            if calculate_errors:
                network.detectors[d].fisher_matrix[k, :, :] = \
                    gw.fishermatrix.FisherMatrix(waveform_model, parameter_values, fisher_parameters, network.detectors[d],once_flag,flag=flag)

            #振幅为信号模
            amp_data=np.abs(signal)
            #相位为信号arctan(虚部/实部)
            phase_data=np.arctan2(np.imag(signal),np.real(signal))
            #保存所有所模拟的有效的信号的振幅和相位信息
            amp_data_all.append(amp_data)
            phase_data_all.append(phase_data)
        network.SNR[k] = np.sqrt(networkSNR_sq)
    gw.detection.analyzeDetections(network, parameters, population, networks_ids)

    if calculate_errors:
        print("begin calcu error")
        gw.fishermatrix.analyzeFisherErrors(network, parameters, fisher_parameters, population, networks_ids)
    ############################################################################################################
    #save all data
    amp_data_all=np.array(amp_data_all)
    phase_data_all=np.array(phase_data_all)

    np.save('amp_data_all_{}.npy'.format(flag),amp_data_all)
    np.save('phase_data_all_{}.npy'.format(flag),phase_data_all)
    if flag[-1]=='0':
        GR_process(flag)
    else:
        wait_for_GR(flag)

    ###########################################################################################################
    #read detector name
    detector_name=''
    for i in range(len(network.detectors)):
        detector_name+=detectors_ids[i]+'_'
    with open('Errors_{}BBH_SNR9.0_{}.txt'.format(detector_name,multiprocessing.current_process().name, 'r')) as f:
        paraname = f.readline().split()
    ###########################################################################################################
    # load fisher data
    Fisherdata = np.load('Fishers_{}BBH_SNR9.0_{}.npy'.format(detector_name, multiprocessing.current_process().name))
    print("Fisherdata", np.shape(Fisherdata))
    #for i in range(0, len(Fisherdata)):
        #np.savetxt("./fisher_data/fisher_{}_{}.csv".format(multiprocessing.current_process().name, i),
                   #Fisherdata[i][:][:], delimiter=',')

    ###########################################################################################################
    #load PE data
    PE_temp=np.loadtxt('Errors_{}BBH_SNR9.0_{}.txt'.format(detector_name,multiprocessing.current_process().name),skiprows=1,unpack=True)
    if PE_temp.ndim == 1:
        PE_temp=PE_temp[:,np.newaxis]
    PE=PE_temp.T
    PE=pd.DataFrame(PE,columns=paraname)
    ###########################################################################################################
    #load invfisher data
    cov_normalizer= np.load('Inv_Fishers_{}BBH_SNR9.0_{}.npy'.format(detector_name,multiprocessing.current_process().name))
    print("cov_normal",np.shape(cov_normalizer))
    #for i in range(0,len(cov_normalizer)):
        #np.savetxt("./cov_normal/inv_fisher_{}_{}.csv".format(multiprocessing.current_process().name,i),cov_normalizer[i][:][:],delimiter=',')
    ###########################################################################################################

    ###########################################################################################################
    cov_mat=[[0 for i in range(0,len(fisher_parameters))] for j in range(0,len(fisher_parameters))]
    for i in range(0,len(fisher_parameters)):
        for j in range(0,len(fisher_parameters)):
            cov_mat[i][j]=np.mean(cov_normalizer[:,i,j])
    cov_mat=np.array(cov_mat)
    #print("cov mat=====",cov_mat)
    ############################################################################################################
    #inv mf-chif fisher data
    fisher_mfchif=Fisherdata[:,-2:,-2:]
    #print("fisher mf chif",np.shape(fisher_mfchif))
    cov_m_chi=[]
    for i in range(0,len(Fisherdata)):
        if np.linalg.det(fisher_mfchif[i])!=0:
            ivfisher_data=np.linalg.inv(fisher_mfchif[i])
            cov_m_chi.append(ivfisher_data)
    cov_m_chi=np.array(cov_m_chi)
    #print("cov_m_chif",cov_m_chi)
    #############################################################################################################


    #calcu systematic errors
    # 即∆thθa
    # load gr data
    data_len=len(amp_data_all)
    #print("data len",data_len)
    amp_gr = np.load('amp_data_all_{}_0.npy'.format(flag[:-2]))
    phase_gr = np.load('phase_data_all_{}_0.npy'.format(flag[:-2]))
    print("amp_gr shape",np.shape(amp_gr))
    print("phase_gr shape", np.shape(phase_gr))
    # load data need calcu error
    amp_theta = np.load('amp_data_all_{}.npy'.format(flag))
    phase_theta = np.load('phase_data_all_{}.npy'.format(flag))
    #print("{}amp_theta{}".format(flag,np.shape(amp_theta)))
    #print("{}phase theta{}".format(flag,np.shape(phase_theta)))
    parameter_gr = pd.read_hdf('./injections/150914_like_population_{}_0.hdf5'.format(flag[:-2]))
    delta_theta_para = [[0 for i in range(len(fisher_parameters))] for j in range(len(cov_normalizer))]
    deriv1_all_data=[]
    deriv2_all_data=[]
    paraGR = [63.1, 0.69]
    #print("valid value",valid_para)
    for k in range(0,len(cov_normalizer)):
        #calcu deriv1
        delta_amp=amp_gr[k,:]-amp_theta[k,:]
        delta_psi=phase_gr[k,:]-phase_theta[k,:]

        deriv1=(delta_amp+1.j*amp_gr[k,:]*delta_psi)*np.exp(1.j*phase_gr[k,:])
        deriv1=np.array(deriv1)
        deriv1_all_data.append(deriv1)
        #deriv1=deriv1[:,np.newaxis]

        #calcu deriv2
        parameter_valuess = parameter_gr.iloc[k]
        #print("parameter values",parameter_values)
        deriv2=[]
        GR_flag=1
        for i in range(0,nd):
            p=fisher_parameters[i]
            deriv2_temp=gw.fishermatrix.derivative(waveform_model,parameter_valuess,p,network.detectors[0],once_flag,GR_flag,amp_gr[k,:],flag)
            deriv2.append(deriv2_temp)
        deriv2_all_data.append(deriv2)

        #calcu delta ij
        delta_ij_sum=0
        for i in range(0,len(fisher_parameters)):
            for j in range(0,len(fisher_parameters)):
                #print("shape deriv1",np.shape(deriv1))
                #print("shape deriv2",np.shape(deriv2))
                ScalarProductForGrTemp=gw.auxiliary.scalar_product(deriv1,deriv2[j][:][:],network.detectors[0])
                #print("{}Scalar Product{}".format(flag,ScalarProductForGrTemp))
                if np.isnan(ScalarProductForGrTemp):
                    print("Nan return")
                    continue
                delta_ij=cov_normalizer[k,i,j]*ScalarProductForGrTemp
                #print("{}cov{}".format(flag,cov_normalizer[k,i,j]))
                #print("{}delta_{}_{}_{}".format(flag,fisher_parameters[i],fisher_parameters[j],delta_ij))
                delta_ij_sum+=delta_ij
            #delta_ij_temp/=float(len(network.detectors)*len(fisher_parameters))
            delta_theta_para[k][i]=delta_ij_sum
            delta_ij_sum=0

    #save deriv data
    deriv2_all_data=np.array(deriv2_all_data)
    deriv1_all_data=np.array(deriv1_all_data)
    deriv1_all_data_save=deriv1_all_data.T
    np.savetxt("deriv1_{}.csv".format(multiprocessing.current_process().name),deriv1_all_data_save[0],delimiter=',')
    print("shape d2 all data", np.shape(deriv2_all_data))
    for i in range(len(fisher_parameters)):
        deriv2_all_data_save=deriv2_all_data[i][:][:][0]
        np.savetxt("deriv2_{}_{}.csv".format(multiprocessing.current_process().name,fisher_parameters[i]),deriv2_all_data_save,delimiter=',')


    #calculation the delta theta of 90% ci
    delta_theta_to_csv=[[0 for i in range(len(fisher_parameters))] for j in range(len(parameters))]
    for i in range(0,len(cov_normalizer)):
        for j in range(len(fisher_parameters)):
            delta_theta_to_csv[i][j]=delta_theta_para[i][j][0]
    delta_theta_to_csv=np.array(delta_theta_to_csv)
    #print("{}delta_theta_to_csv:{}".format(flag,delta_theta_to_csv))
    np.savetxt('delta_theta_para_{}.csv'.format(flag),delta_theta_to_csv,delimiter=',',fmt='%f')
    mid_delta_mf,lo_delta_mf,high_delta_mf=get_ci(delta_theta_to_csv[:,-2])
    mid_delta_chif,lo_delta_chif,high_delta_chif=get_ci(delta_theta_to_csv[:,-1])
    print("{}mid delta mf{}+{}-{}".format(multiprocessing.current_process().name, mid_delta_mf, high_delta_mf - mid_delta_mf,
                                          mid_delta_mf - lo_delta_mf))
    print("{}mid delta chif{}+{}-{}".format(multiprocessing.current_process().name, mid_delta_chif,
                                            high_delta_chif - mid_delta_chif, mid_delta_chif - lo_delta_chif))
    #paraGR = [63.1, 0.69]
    delta_theta_mf_90_ci = [i for i in delta_theta_to_csv[:, -2] if i<high_delta_mf and i>lo_delta_mf]
    delta_theta_chif_90_ci=[i for i in delta_theta_to_csv[:, -1] if i<high_delta_chif and i>lo_delta_chif]
    if np.all(np.isnan(delta_theta_mf_90_ci)):
        mean_delta_chif=0
        mean_delta_mf=0
    else:
        mean_delta_mf=np.mean(delta_theta_mf_90_ci)
        mean_delta_chif=np.mean(delta_theta_chif_90_ci)
        print("{}mean delta mf{}".format(multiprocessing.current_process().name, mean_delta_mf))
        print("{}mean delta chif{}".format(multiprocessing.current_process().name, mean_delta_chif))
    cov_MF_chiF=np.array(cov_mat[-2:,-2:])

    #paraGR=[63.1,0.69]
    paraGR_Mod=paraGR
    if int(flag[-1])!=0:
        paraGR_Mod = [paraGR[0]+mid_delta_mf, paraGR[1]+mid_delta_chif]
        #paraGR_Mod = [max_prob_mf, max_prob_chif]
        #paraGR_Mod=[max_prob_mf_chif[0],max_prob_mf_chif[1]]
        print("current thread{},paraGR_Mod{}".format(multiprocessing.current_process().name, paraGR_Mod))
    plot_para_dic={'pmean':paraGR_Mod,'cov':cov_MF_chiF,'flag':flag,'GR_val':paraGR}
    np.save("plot_para_{}.npy".format(flag),plot_para_dic)
    #plot_Mf_chif(paraGR_Mod,cov_MF_chiF,flag,paraGR)
    #plt.savefig('testMod.png')

    """    
    paramean=[np.mean(PE['lnAGR']),np.mean(PE['phase']),np.mean(PE['eta']),np.mean(PE['chi_s']),paraGR[0]+mid_delta_mf,paraGR[1]+mid_delta_chif]
    var_in_cov_matrix = ['lnAGR', 'phase', 'eta', 'chi_s', 'Mf','af']
    cov_matrix=pd.DataFrame(cov_mat,columns=fisher_parameters,index=fisher_parameters)

    cov_mat_plt_temp=cov_matrix[var_in_cov_matrix]
    cov_mat_plt=cov_mat_plt_temp.loc[var_in_cov_matrix]
    cov_mat_plt=np.array(cov_mat_plt)
    plot_corner(paramean,cov_mat_plt,flag)
    """



if __name__ == '__main__':
    start_time = time.time()
    #multiprocessing.freeze_support()
    flag_1='ins_Modify'
    flag_2='MR_Modify'
    print("init process")
    t1=multiprocessing.Process(target=main,args=(flag_1,),name='ins_main')
    t2=multiprocessing.Process(target=main,args=(flag_2,),name='MR_main')
    print("start process")
    t1.start()
    t2.start()
    print("join")
    t1.join()
    t2.join()
    plot_flag=['ins_Modify_0','ins_Modify_1','ins_Modify_2','ins_Modify_3',
               'MR_Modify_0','MR_Modify_1','MR_Modify_2','MR_Modify_3']
    pmean=[]
    cov=[]
    GR_val=[]
    for i in range(0,len(plot_flag)):
        plot_dic=np.load("plot_para_{}.npy".format(plot_flag[i]),allow_pickle=True).item()
        pmean=plot_dic['pmean']
        cov=plot_dic['cov']
        flag=plot_flag[i]
        GR_val=plot_dic['GR_val']
        plot_Mf_chif(pmean,cov,flag,GR_val)
    plt.savefig('testMod.png')

    #main('ins')
    print("--- %s seconds ---" % (time.time() - start_time))
