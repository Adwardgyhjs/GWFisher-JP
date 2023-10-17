import numpy as np
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
import GWFish.modules.auxiliary as aux
import threading
import multiprocessing
import pandas as pd
import GWFish.modules.constants as cst
from multiprocessing import Lock
lock=Lock()


def invertSVD(matrix):
    thresh = 1e-10
    #print("{}matrix{}".format(threading.current_thread().name,matrix))
    dm = np.sqrt(np.diag(matrix))
    #print("dm",dm)
    normalizer = np.outer(dm, dm)
    #print("{}normalizer{}".format(threading.current_thread().name, normalizer))
    matrix_norm = matrix / normalizer
    #print("{}normalizer{}".format(threading.current_thread().name,normalizer))

    #SVD分解求逆
    [U, S, Vh] = np.linalg.svd(matrix_norm)
    #print("U:{} S:{} Vh:{}".format(U,S,Vh))

    kVal = sum(S > thresh)
    matrix_inverse_norm = U[:, 0:kVal] @ np.diag(1. / S[0:kVal]) @ Vh[0:kVal, :]
    #print("inverse,norm",matrix_inverse_norm/normalizer)
    #print("normalizer",normalizer)
    print("{} SVD分解后得到的逆：{}".format(multiprocessing.current_process().name,matrix_inverse_norm / normalizer))

    #cholesky分解
    F=np.linalg.cholesky(matrix)
    FT=F.T
    FMulFT=np.dot(F,FT)
    print("{} 验证cholesky分解: {}".format(multiprocessing.current_process().name,FMulFT))
    matrix_inverse_cholesky=np.linalg.inv(FT)@ np.linalg.inv(F)
    print("{} cholesky分解后求得的逆：{}".format(multiprocessing.current_process().name,matrix_inverse_cholesky))

    return matrix_inverse_norm / normalizer, S


def derivative(waveform, parameter_values, p, detector,once_flag,GR_flag,amp_GR,flag=None):

    """
    Calculates derivatives with respect to geocent_time, merger phase, and distance analytically.
    Derivatives of other parameters are calculated numerically.
    """

    local_params = parameter_values.copy()
    #print("local para",local_params)
    ff = detector.frequencyvector * cst.G * local_params['M'] / cst.c ** 3
    tc = local_params['geocent_time']
    eta=local_params['eta']
    amp_ins=np.loadtxt("amp_temp_{}.csv".format(flag),dtype='float64')
    ones = np.ones((len(ff), 1))
    Agr=np.exp(local_params['lnAGR'])*ones
    Fisco = np.loadtxt("Fisco_{}.csv".format(flag),dtype='float64')
    ret = np.where(detector.frequencyvector >= Fisco)
    plot_f2 = ret[0][0]
    if GR_flag==1:
        amp_ins = amp_GR
    if p == 'luminosity_distance' :
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector,once_flag,flag=flag)
        derivative = -1. / local_params[p] * det.projection(local_params, detector, wave, t_of_f)
    elif p == 'geocent_time':
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector,once_flag,flag=flag)
        derivative = 2j * np.pi * detector.frequencyvector * det.projection(local_params, detector, wave, t_of_f)
    elif p == 'phase':
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector,once_flag,flag=flag)
        derivative = -1j * det.projection(local_params, detector, wave, t_of_f)
    elif p=='lnAGR':
        wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector, once_flag, flag=flag)
        derivative= det.projection(local_params, detector, wave, t_of_f)
    else:
        pv = local_params[p]
        eps = 1e-5 # this follows the simple "cube root of numerical precision" recommendation, which is 1e-16 for double
        dp = np.maximum(eps, eps * pv)

        pv_set1 = parameter_values.copy()
        pv_set2 = parameter_values.copy()

        pv_set1[p] = pv - dp / 2.
        pv_set2[p] = pv + dp / 2.

        if p in ['ra', 'dec', 'psi']:  # these parameters do not influence the waveform
            wave, t_of_f = wf.hphc_amplitudes(waveform, local_params, detector.frequencyvector,once_flag,flag=flag)

            signal1 = det.projection(pv_set1, detector, wave, t_of_f)
            signal2 = det.projection(pv_set2, detector, wave, t_of_f)
            derivative = (signal2 - signal1) / dp
        else:
            pv_set1['geocent_time'] = 0.  # to improve precision of numerical differentiation
            pv_set2['geocent_time'] = 0.
            wave1, t_of_f1 = wf.hphc_amplitudes(waveform, pv_set1, detector.frequencyvector,once_flag,flag=flag)
            wave2, t_of_f2 = wf.hphc_amplitudes(waveform, pv_set2, detector.frequencyvector,once_flag,flag=flag)
 
            pv_set1['geocent_time'] = tc
            pv_set2['geocent_time'] = tc
            signal1 = det.projection(pv_set1, detector, wave1, t_of_f1+tc)
            signal2 = det.projection(pv_set2, detector, wave2, t_of_f2+tc)

            derivative = np.exp(2j * np.pi * detector.frequencyvector * tc) * (signal2 - signal1) / dp

    # print(fisher_parameters[p] + ': ' + str(derivative))
    return derivative


def FisherMatrix(waveform, parameter_values, fisher_parameters, detector,once_flag,flag=None):

    nd = len(fisher_parameters)
    fm = np.zeros((nd, nd))
    sigma = pd.read_csv('sigma_{}.csv'.format(flag))
    for p1 in np.arange(nd):
        deriv1_p = fisher_parameters[p1]
        deriv1 = derivative(waveform, parameter_values, deriv1_p, detector,once_flag,0,0,flag=flag)
        if sigma[fisher_parameters[p1]][0]==0:
            fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, detector), axis=0)
        # sum Fisher matrices from different components of same detector (e.g., in the case of ET)
        else:
            fm[p1, p1] = np.sum(aux.scalar_product(deriv1, deriv1, detector), axis=0)+1./(sigma[fisher_parameters[p1]][0]**2)
        for p2 in np.arange(p1+1, nd):
            deriv2_p = fisher_parameters[p2]
            deriv2 = derivative(waveform, parameter_values, deriv2_p, detector,once_flag,0,0,flag=flag)
            fm[p1, p2] = np.sum(aux.scalar_product(deriv1, deriv2, detector), axis=0)
            fm[p2, p1] = fm[p1, p2]
    return fm


def analyzeFisherErrors(network, parameter_values, fisher_parameters, population, networks_ids):
    """
    Analyze parameter errors.
    """

    # Check if sky-location parameters are part of Fisher analysis. If yes, sky-location error will be calculated.
    signals_havesky = False
    if ('ra' in fisher_parameters) and ('dec' in fisher_parameters):
        signals_havesky = True
        i_ra = fisher_parameters.index('ra')
        i_dec = fisher_parameters.index('dec')
    signals_haveids = False
    if 'id' in parameter_values.columns:
        signals_haveids = True
        signal_ids = parameter_values['id']
        parameter_values.drop('id', inplace=True, axis=1)


    npar = len(fisher_parameters)
    ns = len(network.detectors[0].fisher_matrix[:, 0, 0])  # number of signals
    N = len(networks_ids)

    detect_SNR = network.detection_SNR

    network_names = []
    for n in np.arange(N):
        network_names.append('_'.join([network.detectors[k].name for k in networks_ids[n]]))

    for n in np.arange(N):
        parameter_errors = np.zeros((ns, npar))
        sky_localization = np.zeros((ns,))
        networkSNR = np.zeros((ns,))
        fishers = np.zeros((ns, npar, npar))
        inv_fishers = np.zeros((ns, npar, npar))
        sing_values = np.zeros((ns, npar))
        
        for d in networks_ids[n]:
            #print("{} network.detectors[d].SNR {}".format(threading.current_thread().name, network.detectors[d].SNR))
            networkSNR += network.detectors[d].SNR ** 2
        networkSNR = np.sqrt(networkSNR)

        for k in np.arange(ns):
            network_fisher_matrix = np.zeros((npar, npar))
            if networkSNR[k] > detect_SNR[1]:
                for d in networks_ids[n]:
                    if network.detectors[d].SNR[k] > detect_SNR[0]:
                        network_fisher_matrix += np.squeeze(network.detectors[d].fisher_matrix[k, :, :])

                if npar > 0 :
                    network_fisher_inverse, S = invertSVD(network_fisher_matrix)
                    fishers[k, :, :] = network_fisher_matrix
                    inv_fishers[k, :, :] = network_fisher_inverse
                    sing_values[k, :] = S
                    parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))

                    if signals_havesky:
                        sky_localization[k] = np.pi * np.abs(np.cos(parameter_values['dec'].iloc[k])) \
                                              * np.sqrt(network_fisher_inverse[i_ra, i_ra]*network_fisher_inverse[i_dec, i_dec]
                                                        -network_fisher_inverse[i_ra, i_dec]**2)
        delim = " "
        header = 'network_SNR '+delim.join(parameter_values.keys())+" "+delim.join(["err_" + x for x in fisher_parameters])
        ii = np.where(networkSNR > detect_SNR[1])[0]

        save_data = np.c_[networkSNR[ii], parameter_values.iloc[ii], parameter_errors[ii, :]]
        #print("{} save data{}".format(threading.current_thread().name, save_data))
        fishers = fishers[ii, :, :]
        inv_fishers = inv_fishers[ii, :, :]
        sing_values = sing_values[ii, :]
        print("current process in save fisher matrix",multiprocessing.current_process().name)
        np.save('Fishers_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1])+'_'+multiprocessing.current_process().name + '.npy', fishers)
        np.save('Inv_Fishers_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1])+'_'+multiprocessing.current_process().name + '.npy', inv_fishers)
        np.save('Sing_Values_'+ network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1])+'_'+multiprocessing.current_process().name + '.npy', sing_values)
        
        if signals_havesky:
            header += " err_sky_location"
            save_data = np.c_[save_data, sky_localization[ii]]
        if signals_haveids:
            header = "signal "+header
            save_data = np.c_[signal_ids.iloc[ii], save_data]

        file_name = 'Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) +'_'+ multiprocessing.current_process().name+'.txt'

        if signals_haveids and (len(save_data) > 0):
            np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1]) +'_'+ multiprocessing.current_process().name+'.txt',
                       save_data, delimiter=' ', fmt='%s ' + "%.3E " * (len(save_data[0, :]) - 1), header=header, comments='')
        else:
            np.savetxt('Errors_' + network_names[n] + '_' + population + '_SNR' + str(detect_SNR[1])+'_' + multiprocessing.current_process().name+'.txt',
                       save_data, delimiter=' ', fmt='%s ' + "%.3E " * (len(save_data[0, :]) - 1), header=header, comments='')

