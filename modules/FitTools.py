#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:21:17 2021

@author: matteo
"""

import numpy as np
import numba
from numba import njit
import h5py
import os


def load_HDF_aligned(h5path,keys):
    
    data_dict = {}
            
    with h5py.File(h5path, 'r') as f0:
        
        for key in keys:
            
            if key=='n_SDD':
                ch_keys = list(f0['dante'].keys())
                
                data = 0
                for k in ch_keys:
                    if k.count('channel') > 0 and k != 'channel_SUM':
                        data += 1
                        
                label = 'n_SDD'
                
            
            if key.count('channel') > 0:                                     
                data = np.asarray(f0['dante/' + key],dtype=np.int64)   
                label = 'xrfdata'
                
                
            if key.count('offset') > 0:
                data = np.float64(f0['dante/' + key])
                label = 'offset'
                
            if key.count('slope') > 0:
                data = np.float64(f0['dante/' + key])
                label = 'slope'
                
            if key.count('mask') > 0:
                data = np.asarray(f0['dante/' + key],dtype=bool)
                label = 'mask'
                
            if key.count('beam_en') > 0:
                data = np.float64(f0['dante/' + key])
                label = 'beam_en'
                
            if key.count('im_shape') > 0:
                data = np.asarray(f0['dante/' + key],dtype=np.int32)
                label = 'im_shape'
                
            if key.count('nonzeroxrf') > 0:
                try:
                    data = np.asarray(f0['dante/' + key],dtype=np.int64)
                    
                except:
                    im_shape = np.asarray(f0['dante/im_shape'],dtype=np.int64)
                    data = np.array([idx for idx in range(im_shape[0]*im_shape[1])],dtype=np.int64)

                label = 'nonzeroxrf'
                
            # if key.count('en_axis') > 0:
            #     data = np.asarray(f0['dante/' + key],dtype=float)   
            #     label = 'xdata'
                
            data_dict.update({label:data})
                            
    return data_dict


def load_HDF_aligned_singlePixel(h5path,key,lin_idxs):
                        
    with h5py.File(h5path, 'r') as f0:
        data = np.asarray(f0['dante/' + key][lin_idxs[0]],dtype=np.int64)
        
        for idx in range(1,len(lin_idxs)):
            data += np.asarray(f0['dante/' + key][lin_idxs[idx]],dtype=np.int64)
                
    return data


def save_tmp_h5_FITfile(data_fld,fit_data):
    
    import uuid
    
    if np.invert(os.path.isdir(data_fld)):
        os.makedirs(data_fld)
        
    fname = str(uuid.uuid4().hex) + '.h5'
    
    out_path = os.path.join(data_fld,fname)
            
    with h5py.File(out_path, 'w') as f0:
        f0.create_dataset('XRF_Fits/auc_map',data=fit_data['auc_map'],compression='gzip',dtype=np.int64)
        
    return out_path

        
def load_tmp_h5_FITfile(out_path,idx):
                
    with h5py.File(out_path, 'r') as f0:
        auc_map = np.asarray(f0['XRF_Fits/auc_map'][:,:,idx],int)

    return auc_map


def load_tmp_h5_FITfile_ext(out_path):
                
    with h5py.File(out_path, 'r') as f0:
        auc_map = np.asarray(f0['XRF_Fits/auc_map'],int)

    return auc_map


def create_custom_HDF5_input(beam_en,xrfdata,im_shape,offset,slope):
    
    out_path=os.path.join(os.getcwd(),'data/input/','xrf_flat_scan_aligned_2fit.h5')
    
    with h5py.File(out_path, 'w') as f0:
        f0.create_dataset('dante/beam_en',data=beam_en,dtype=np.float64)
        f0.create_dataset('dante/channel_SUM',data=xrfdata,compression='gzip',dtype=np.int64)
        f0.create_dataset('dante/im_shape',data=im_shape,dtype=np.int32)
        f0.create_dataset('dante/offset',data=offset,dtype=np.float64)
        f0.create_dataset('dante/slope',data=slope,dtype=np.float64)


def auc2element(auc_map,element_list,lines_list,scattering_flag):
    
    # initialisations
    if scattering_flag:
        dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],len(element_list) + 1),int)
        
    else:
        dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],len(element_list)),int)
    
    idx_peak = 0
    for idx_el in range(len(element_list)):        
        for idx_fam in range(len(lines_list[idx_el])):
            for idx_line in range(len(lines_list[idx_el][idx_fam])):
                dat[:,:,idx_el] += auc_map[:,:,idx_peak]
                idx_peak += 1                        
                
        
    if scattering_flag:
        dat[:,:,-1] = auc_map[:,:,-1]
        
    return dat


def organise_folder_structure():
    
    
    tmp_fld = os.path.join(os.getcwd(),'data/tmp/')
    res_fld = os.path.join(os.getcwd(),'data/results/')
    input_fld = os.path.join(os.getcwd(),'data/input/')
    
    if np.invert(os.path.isdir(tmp_fld)):
        os.makedirs(tmp_fld)
        
    if np.invert(os.path.isdir(res_fld)):
        os.makedirs(res_fld)
        
    if np.invert(os.path.isdir(input_fld)):
        os.makedirs(input_fld)
        
    return tmp_fld, res_fld, input_fld
            
                
def save_auc_elements_2_h5_allSDDs(h5path,element_list,auc_map,scattering_flag):
    
    fname_base = os.path.splitext(os.path.split(h5path)[1])[0]
    
    if auc_map.ndim < 4:
        
        with h5py.File(h5path, "w") as f:   
            
            for idx_el in range(len(element_list)):
                
                fname = fname_base + '/xrf_fit/results/' + element_list[idx_el]
                f.create_dataset(fname,data=auc_map[:,:,idx_el]  ,dtype=np.int64,compression='gzip')    

                
            if scattering_flag:
                fname = fname_base + '/xrf_fit/results/Scattering'
                f.create_dataset(fname,data=auc_map[:,:,-1].astype(int).copy(),dtype=np.int64,compression='gzip')
        
    else:
        
        with h5py.File(h5path, "w") as f:   
            for idx_el in range(len(element_list)):
                
                fname = fname_base + '/xrf_fit/results/' + element_list[idx_el]
                dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
                
                for idx_SDD in range(np.shape(auc_map)[2]):
                    dat[:,:,idx_SDD] += auc_map[:,:,idx_SDD,idx_el]
                        
                f.create_dataset(fname,data=dat,dtype='int32',compression='gzip')        
                
            if scattering_flag:
                fname = fname_base + '/xrf_fit/results/Scattering'
                dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
                for idx_SDD in range(np.shape(auc_map)[2]):
                    dat[:,:,idx_SDD] = auc_map[:,:,idx_SDD,-1]
                            
                f.create_dataset(fname,data=dat,dtype=np.int64,compression='gzip')  
                
                
def save_auc_lines_2_h5_allSDDs(h5path,element_list,lines_list,auc_map,scattering_flag):
    
    fname_base = os.path.splitext(os.path.split(h5path)[1])[0]
    
    if auc_map.ndim < 4:
        
        with h5py.File(h5path, "w") as f:   
            
            idx_line = 0
            
            for idx_el in range(len(element_list)):
                for idx_fam in range(len(lines_list[idx_el])):
                    for idx in range(len(lines_list[idx_el][idx_fam])):
                
                        fname = fname_base + '/xrf_fit/results/' + element_list[idx_el] + '_' + lines_list[idx_el][idx_fam][idx]
                        f.create_dataset(fname,data=auc_map[:,:,idx_line]  ,dtype=np.int64,compression='gzip')    
                        idx_line += 1
                
            if scattering_flag:
                fname = fname_base + '/xrf_fit/results/Scattering'
                f.create_dataset(fname,data=auc_map[:,:,-1].astype(int).copy(),dtype=np.int64,compression='gzip')
        
    else:
        
        with h5py.File(h5path, "w") as f: 
            
            idx_line = 0
            for idx_el in range(len(element_list)):
                for idx_fam in range(len(lines_list[idx_el])):
                    for idx in range(len(lines_list[idx_el][idx_fam])):
                
                        fname = fname_base + '/xrf_fit/results/' + element_list[idx_el] + '_' + lines_list[idx_el][idx_fam][idx]
                        dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
                        
                        for idx_SDD in range(np.shape(auc_map)[2]):
                            dat[:,:,idx_SDD] += auc_map[:,:,idx_SDD,idx_line]
                                
                        f.create_dataset(fname,data=dat,dtype=np.int64,compression='gzip')        
                        idx_line += 1
                    
            if scattering_flag:
                fname = fname_base + '/xrf_fit/results/Scattering'
                dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
                for idx_SDD in range(np.shape(auc_map)[2]):
                    dat[:,:,idx_SDD] = auc_map[:,:,idx_SDD,-1]
                            
                f.create_dataset(fname,data=dat,dtype=np.int64,compression='gzip')  
            
            
def save_auc_2_tiff_allSDDs(tiffpath,element_list,auc_map,scattering_flag):
    
    import tifffile
    
    # fname_base = os.path.splitext(os.path.split(tiffpath)[1])[0]
    path = os.path.splitext(os.path.split(tiffpath)[0])[0]

    if auc_map.ndim < 4:
        
        # with tifffile.TiffWriter(tiffpath, bigtiff=False) as f:   
        for idx_el in range(len(element_list)):
            
            dat = auc_map[:,:,idx_el]  
            tifffile.imwrite(path + '/' + element_list[idx_el] + '_XRFitProc.tiff',dat.astype(np.float64))
            
        if scattering_flag:
            tifffile.imwrite(path + '/' + 'Scattering_XRFitProc.tiff',auc_map[:,:,-1].astype(np.float64))
        
    else:
    
        
        for idx_el in range(len(element_list)):
            
            dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
            
            for idx_SDD in range(np.shape(auc_map)[2]):
                dat[:,:,idx_SDD] += auc_map[:,:,idx_SDD,idx_el]
           
            tifffile.imwrite(path + '/' + element_list[idx_el] + '_XRFitProc.tiff',data=dat.astype(np.float64))
            
        if scattering_flag:
            dat = np.zeros((np.shape(auc_map)[0],np.shape(auc_map)[1],np.shape(auc_map)[2]),int)
            for idx_SDD in range(np.shape(auc_map)[2]):
                dat[:,:,idx_SDD] = auc_map[:,:,idx_SDD,-1]
                        
            tifffile.imwrite(path + '/' + 'Scattering_XRFitProc.tiff',auc_map[:,:,-1].astype(np.float64))


def sort_user_lines(element_list,xrf_flat_list,beam_energy,en_bounds,thr_line_strength,flag_double_energies):
    
    import xraydb
    
    if not element_list:
        return [], [], np.array([0],float), np.array([0],float), np.array([0],float)
    
    xrdb = xraydb.XrayDB()
    
    # initialisations            
    lines_intensity_arr = []
    lines_strength_arr = []
    lines_energy_arr = []
    lines_list_tmp = []    
    
    xfr_flat_arr = []
    element_arr = []
    for idx in range(len(xrf_flat_list)):
        # line = xrf_flat_list[idx][np.int64(xrf_flat_list[idx].find(' '))+3:]
        el = xrf_flat_list[idx][:np.int64(xrf_flat_list[idx].find(' '))]
        line = xrf_flat_list[idx][np.int64(xrf_flat_list[idx].find('-'))+2:np.int64(xrf_flat_list[idx].rfind('-'))-1]
        xfr_flat_arr.append(line)
        element_arr.append(el)
        
    element_arr = np.asarray(element_arr)    
    xfr_flat_arr = np.asarray(xfr_flat_arr)    
        
    if el not in element_list:
        element_list.append(el)  
    
    for idx_el in range(len(element_list)):
                        
        tmp_fam_lines_label = []
        lines_dict = xraydb.xray_lines(element_list[idx_el],excitation_energy=beam_energy)
                        
        tmp_lines_keys = list(lines_dict.keys())

        for idx_line in range(len(tmp_lines_keys)):
            
            # tmp_line_label = []
            tmp_line_label = ''
            
            # condition on fit energy boundaries
            if lines_dict[tmp_lines_keys[idx_line]][0] > en_bounds[0] and lines_dict[tmp_lines_keys[idx_line]][0] < en_bounds[1]:
                                            
                # tmp_line_label.append(tmp_lines_keys[idx_line])
                tmp_line_label = tmp_lines_keys[idx_line]
                
                if len(tmp_line_label) > 0:
                    tmp_fam_lines_label.append([tmp_line_label,xraydb.xray_line(element_list[idx_el],tmp_line_label)[2]])

        if len(tmp_fam_lines_label)==0:
            raise ValueError('\nERROR: element '+element_list[idx_el]+' is not within fit boundaries, remove it from element list, or try adjusting the boundaries\n')
            
        lines_list_tmp.append(tmp_fam_lines_label)
        
        
    # sort lines in families
    lines_list_all = []
    for idx_el in range(len(element_list)):
        
        fam_lines = []
        
        lines_in_el = np.asarray(lines_list_tmp[idx_el])
        init_lvs = np.unique(lines_in_el[:,1])
                
        for lv in init_lvs:
            lv_idxs = np.argwhere(lines_in_el==lv)[:,0]
            fam_lines.append([lines_in_el[idx,0] for idx in lv_idxs])
        
        lines_list_all.append(fam_lines)
        
    
    # compare the list of all possible lines for a given energy VS the lines the user wants to use
    lines_list = []
    for idx_el in range(len(lines_list_all)):
        fam_tmp = []
        lines_in_el = xfr_flat_arr[np.argwhere(element_arr==element_list[idx_el])[:,0]]
        
        for idx_fam in range(len(lines_list_all[idx_el])):
            lines_tmp = []
            for idx_flat_line in range(len(lines_in_el)):
                if lines_in_el[idx_flat_line] in lines_list_all[idx_el][idx_fam]:
                    if lines_in_el[idx_flat_line] not in lines_tmp:
                        lines_tmp.append(lines_in_el[idx_flat_line])
                    
            fam_tmp.append(lines_tmp)
            
        lines_list.append(fam_tmp)    
        
        
        
    for idx_el in range(len(element_list)):
        lines_strenghts_dict = xrdb.xray_line_strengths(element_list[idx_el],excitation_energy=beam_energy)  
        lines_dict = xraydb.xray_lines(element_list[idx_el],excitation_energy=beam_energy)
        tmp_lines_keys = []
        
        for idx_fam in range(len(lines_list[idx_el])):
            for idx_line in range(len(lines_list[idx_el][idx_fam])):
                tmp_lines_keys.append(lines_list[idx_el][idx_fam][idx_line])        

        for idx_line in range(len(tmp_lines_keys)):            
            tmp_line_label = ''
               
            # condition on fit energy boundaries
            if lines_dict[tmp_lines_keys[idx_line]][0] > en_bounds[0] and lines_dict[tmp_lines_keys[idx_line]][0] < en_bounds[1]:
                lines_intensity_arr.append(lines_dict[tmp_lines_keys[idx_line]][1])
                lines_energy_arr.append(lines_dict[tmp_lines_keys[idx_line]][0])
                lines_strength_arr.append(lines_strenghts_dict[tmp_lines_keys[idx_line]])
        
    return lines_list_all, lines_list, np.asarray(lines_energy_arr,float), np.asarray(lines_intensity_arr,float), np.asarray(lines_strength_arr,float)


def find_strongest_line_in_family(element_list, lines_list, beam_en):
    
    import xraydb
    
    xrdb = xraydb.XrayDB()
    
    # search for lines within delta_en
    idx = 0
    best_lines_list = []
    lin_coords_lines = []
    for idx_el in range(len(lines_list)):
        
        best_line_idx = np.zeros((len(lines_list[idx_el]),),int)
        
        for idx_fam in range(len(lines_list[idx_el])):
            
            nn_str = np.zeros((len(lines_list[idx_el][idx_fam]),))
            
            tmp_lin_idxs = np.zeros((len(lines_list[idx_el][idx_fam]),),int)
            # lin_idxs = np.zeros((len(lines_list[idx_el][idx_fam]),),int)
            
            for idx_line in range(len(lines_list[idx_el][idx_fam])):
                
                nn_str[idx_line] = xrdb.xray_line_strengths(element_list[idx_el],beam_en)[lines_list[idx_el][idx_fam][idx_line]]
                                
                # save linear index corresponding to each line
                tmp_lin_idxs[idx_line] = idx
                
                idx += 1
                
            # find strongest line for family in exam
            tmp_best_idx = np.argmax(nn_str)
            best_line_idx[idx_fam] = tmp_best_idx
            
            # place the lin cooord of the strongest line first and then fill in the others            
            for idx_lin_idxs in range(len(tmp_lin_idxs)): 
                lin_coords_lines.append([tmp_lin_idxs[tmp_best_idx],tmp_lin_idxs[idx_lin_idxs]])
                        
        best_lines_list.append([[lines_list[idx_el][idx_fam][best_line_idx[idx_fam]]] for idx_fam in range(len(lines_list[idx_el]))])
    
    return best_lines_list, np.asarray(lin_coords_lines,int)
    

@njit
def snip_bkg(ydat, snip_width, n_iter):
    
    # inititialisations
    w = np.zeros((np.shape(ydat)[0],),numba.float64)
    v = ydat.copy()   
        
    for _ in range(n_iter): 
        for idx_iter in range(snip_width):
                
            for idx_ch1 in range(idx_iter,np.shape(ydat)[0]-idx_iter+1):
    
                a1 = v[idx_ch1]
                a2 = (v[idx_ch1 - idx_iter] + v[idx_ch1 + idx_iter])/2
                w[idx_ch1] = min(a1,a2) 
             
            for idx_ch2 in range(idx_iter,np.shape(ydat)[0]-idx_iter):
                v[idx_ch2] = w[idx_ch2]
                
    return v


@njit
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-np.power(x - mean,2) / (2*np.power(sigma,2)))


@njit
def minimise_model(init_pars,*args):
    
    xdata = args[0]
    ydata = args[1]
    peaks_center = args[2]
    
    ytmp = np.zeros(np.shape(ydata)[0],np.float64)
    idx_peak = 0
    for idx in range(0,len(init_pars),2):
        ytmp += gaussian(xdata,init_pars[idx],peaks_center[idx_peak],init_pars[idx+1])            
        idx_peak += 1    
        
    lsqd = 0.0
    for idx in range(np.shape(ydata)[0]):
        lsqd += (ytmp[idx] - ydata[idx])**2
    
    return np.sqrt(lsqd)


@njit
def model_func_sum_components(x, peaks_center, *params):
    
    n_peaks = len(params)//2
    peaks = np.zeros((n_peaks,np.shape(x)[0]))
    idx_peak = 0
    for idx in range(0,len(params),2):
        peaks[idx_peak,:] += gaussian(x,params[idx],peaks_center[idx_peak],params[idx+1])
        idx_peak += 1
        
    return peaks


@njit
def correct_decimal_spectrum(ydat):
    
    ydat_corr = ydat.copy()
    
    for idx_ch in range(np.shape(ydat)[0]):
        
        if np.isnan(ydat[idx_ch]):
            ydat[idx_ch] = 0
        
        if ydat[idx_ch] < 1:
            ydat_corr[idx_ch] = 0
            
    return ydat_corr


@njit(cache=True)
def calc_init_params_scattering(ydata,bkg,peak_centers_ch,sigma_bounds,n_lines,line_int,lin_coords,beam_en_ch,delta_ch):
        
    # initialisations
    init_params = np.zeros(((n_lines + 1)*2,))
    lb = np.zeros(((n_lines + 1)*2,))
    ub = np.zeros(((n_lines + 1)*2,))    

    # add init params for all line peaks
    idx_count = 0
    for idx in range(n_lines):
        
        # take the amplitde value at the peak maximum
        if peak_centers_ch[lin_coords[idx,1]] - delta_ch >= 0 and peak_centers_ch[lin_coords[idx,1]] + delta_ch <= np.shape(ydata)[0]:
            amp_best_max_norm = np.max(ydata[peak_centers_ch[lin_coords[idx,1]]-delta_ch : peak_centers_ch[lin_coords[idx,1]]+delta_ch] - bkg[peak_centers_ch[lin_coords[idx,1]]-delta_ch : peak_centers_ch[lin_coords[idx,1]]+delta_ch])
            
        else:
            amp_best_max_norm = (ydata[peak_centers_ch[lin_coords[idx,1]]] - bkg[peak_centers_ch[lin_coords[idx,1]]])
            
            
        amp_max = amp_best_max_norm*line_int[lin_coords[idx,1]]
                                           
        init_params[idx_count:idx_count+2] = [amp_max, (sigma_bounds[1]+sigma_bounds[0])/2]
        lb[idx_count:idx_count+2] = [0, sigma_bounds[0]]
        ub[idx_count:idx_count+2] = [amp_max + 1, sigma_bounds[1]]
        
        idx_count += 2
        
    # add init params for scattering peak
    amp_max = np.max(ydata[beam_en_ch-delta_ch:beam_en_ch+delta_ch] - bkg[beam_en_ch-delta_ch : beam_en_ch+delta_ch])
    init_params[-2:] = [amp_max, (sigma_bounds[1] + sigma_bounds[0])/2]
    lb[-2:] = [0, sigma_bounds[0]]
    ub[-2:] = [amp_max + 1, sigma_bounds[1]]
                
    return init_params, lb, ub 


@njit(parallel=True,cache=True)
def fitparams2auc_flat(fit_params,peaks_center,xdata,nsigma):
    
    # initialisations
    n_peaks = np.shape(fit_params)[1]//2
    auc_map = np.zeros((np.shape(fit_params)[0],np.shape(fit_params)[2],n_peaks),np.int64) #numba.int64
    
    for idx_sdd in numba.prange(np.shape(fit_params)[2]):
        for idx_coord in range(np.shape(fit_params)[0]):
            params = fit_params[idx_coord,:,idx_sdd]
            
            idx_peak = 0
            for idx in range(0,len(params),2):
                line_mask = np.zeros((len(xdata),),np.bool_)
                
                for idx_x in range(len(xdata)):
                    if xdata[idx_x] >= peaks_center[idx_peak] - nsigma*params[idx+1] and xdata[idx_x] <= peaks_center[idx_peak] + nsigma*params[idx+1]:
                        line_mask[idx_x] = 1
                
                auc_map[idx_coord,idx_sdd,idx_peak] += np.sum((correct_decimal_spectrum(gaussian(xdata,params[idx],peaks_center[idx_peak],params[idx+1])))*line_mask)
                idx_peak += 1

    return auc_map


@njit(parallel=True, cache=True)
def fitparams2auc_flat_SDDsum(fit_params,peaks_center,xdata,nsigma):
    
    # initialisations
    n_peaks = np.shape(fit_params)[1]//2
    auc_map = np.zeros((np.shape(fit_params)[0],n_peaks),np.int64) #numba.int64
    
    for idx_coord in numba.prange(np.shape(fit_params)[0]):
        params = fit_params[idx_coord,:]
        
        idx_peak = 0
        for idx in range(0,len(params),2):
            
            line_mask = np.zeros((len(xdata),),numba.boolean) #numba.boolean
            
            for idx_x in range(len(xdata)):
                if xdata[idx_x] >= peaks_center[idx_peak] - nsigma*params[idx+1] and xdata[idx_x] <= peaks_center[idx_peak] + nsigma*params[idx+1]:
                    line_mask[idx_x] = 1
                    
            
            tmp = np.sum((gaussian(xdata,params[idx],peaks_center[idx_peak],params[idx+1]))*line_mask)            
            if np.isnan(tmp):
                tmp = 0.0
            
            
            auc_map[idx_coord,idx_peak] += tmp
            idx_peak += 1

    return auc_map


def fit_chunk_scattering(data_flat,info_dict,idx_wrk):
    
    # from scipy.signal import savgol_filter
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    
    # initialisations
    fit_params = np.zeros((np.shape(data_flat)[0],(info_dict['n_lines'] + 1)*2),dtype='float32')
    sigma_bounds = tuple(info_dict['sigma_bounds'])
    peak_centers_ch = np.asarray(info_dict['peak_centers_ch'],int)
    lines_intensity = np.asarray(info_dict['lines_intensity'],float)
    lin_coords_lines = np.asarray(info_dict['lin_coords_lines'],int)
    xdata = np.asarray(info_dict['xdata'],float)
    lines_energy_scattering = np.asarray(info_dict['lines_energy_scattering'],float)
    
    n_lines = np.int64(info_dict['n_lines'])
    beam_en_ch = np.int64(info_dict['beam_en_ch'])
    delta_peak = np.int64(info_dict['delta_peak'])
    fit_method = str(info_dict['OPT_method'])
    ftol = np.float64(info_dict['ftol'])
    nfev = np.int64(info_dict['nfev'])
    
    for idx_coord in range(np.shape(data_flat)[0]):         
        ydata_vox_filt = data_flat[idx_coord,:]
        
        if np.max(ydata_vox_filt) > 0:
            ratio = np.max(data_flat[idx_coord,:])/np.max(ydata_vox_filt)
            ydata_vox_filt = ydata_vox_filt*ratio
            
            # ydata_vox_filt = data_flat[idx_coord,:].copy()
            bkg_vox = snip_bkg(ydata_vox_filt,info_dict['snip_width'],info_dict['snip_iter'])
            
            init_params_vox, lb_vox, ub_vox = calc_init_params_scattering(ydata_vox_filt,bkg_vox,peak_centers_ch,sigma_bounds,n_lines,lines_intensity,lin_coords_lines,beam_en_ch,delta_peak)
            
            bounds_tmp = Bounds(lb_vox,ub_vox)
            line_params = minimize(minimise_model,init_params_vox,args=(xdata, ydata_vox_filt - bkg_vox, lines_energy_scattering), method=fit_method, options={'ftol':ftol, 'maxiter':nfev}, bounds=bounds_tmp)
            
            fit_params[idx_coord,:] = line_params['x']
            
        else:
            fit_params[idx_coord,:] = 0.0
        
    return [fit_params, idx_wrk]


def fit_pixel(data_flat,xdata,info_dict):
    
    # from scipy.signal import savgol_filter
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    
    fit_bounds = tuple(info_dict['fit_bounds'])
    
    ydata_vox_filt = correct_decimal_spectrum(data_flat)[fit_bounds[0]:fit_bounds[1]]
    bkg_vox = snip_bkg(ydata_vox_filt,np.float64(np.float64(info_dict['snip_width'])),np.float64(info_dict['snip_iter']))
    
    ratio = np.max(data_flat)/np.max(ydata_vox_filt)
    ydata_vox_filt = ydata_vox_filt*ratio
    
    init_params_vox, lb_vox, ub_vox = calc_init_params_scattering(ydata_vox_filt,bkg_vox,np.asarray(info_dict['peak_centers_ch'],int),np.asarray(info_dict['sigma_bounds']),np.int64(info_dict['n_lines']),np.asarray(info_dict['lines_intensity'],float),np.asarray(info_dict['lin_coords_lines'],int),np.int64(info_dict['beam_en_ch']),np.int64(info_dict['delta_peak']))
    
    bounds_tmp = Bounds(lb_vox,ub_vox)
    line_params = minimize(minimise_model,init_params_vox,args=(xdata[fit_bounds[0]:fit_bounds[1]], ydata_vox_filt - bkg_vox, np.asarray(info_dict['lines_energy_scattering'],float)),method=str(info_dict['OPT_method']),options={'ftol':0.001, 'maxiter':np.int64(info_dict['nfev'])},bounds=bounds_tmp)
        
    fits = np.zeros((np.shape(xdata)[0],),float)
    fits[fit_bounds[0]:fit_bounds[1]] = np.sum(model_func_sum_components(xdata[fit_bounds[0]:fit_bounds[1]],np.asarray(info_dict['lines_energy_scattering'],float),*line_params['x']),0)
        
    return fits, ydata_vox_filt


def save_fit_info_2_txt(txtpath,info_dict):
    
    # write run info to file
    txtfile = open(txtpath, 'w')
    
    # write elements info
    txtfile.write('# elements and lines fitted\n')
    idx = 0
    for idx_el in range(len(info_dict['element_list'])):
        txtfile.write(info_dict['element_list'][idx_el] +' = ')
        
        for idx_fam in range(len(info_dict['lines_list_user'][idx_el])):
            for idx_line in range(len(info_dict['lines_list_user'][idx_el][idx_fam])):
                
                if idx_line != len(info_dict['lines_list_user'][idx_el][idx_fam]) - 1:
                    txtfile.write(str(info_dict['lines_list_user'][idx_el][idx_fam][idx_line])+' ('+str(info_dict['lines_energy'][idx])+'), ')
                    
                else:
                    txtfile.write(str(info_dict['lines_list_user'][idx_el][idx_fam][idx_line])+' ('+str(info_dict['lines_energy'][idx])+')')
                    
                idx += 1
    
        txtfile.write('\n')
    
    txtfile.write('\n')
    
    # write incident energy
    txtfile.write('# incident beam energy\n')
    txtfile.write('beam_en = '+str(info_dict['beam_en'])+'\n\n')
    
    # write calibration parameters
    txtfile.write('# calibration parameters\n')
    txtfile.write('slope = '+str(info_dict['slope'])+'\n'+'offset = '+str(info_dict['offset'])+'\n\n') 
    
    # write fit boundaries parameters
    txtfile.write('# fit boundaries parameters\n')
    txtfile.write('channels_lb = '+str(info_dict['fit_bounds'][0])+'\n'+'channels_ub = '+str(info_dict['fit_bounds'][1])+'\n\n') 
    
    # # write Savitzky-Golay parameters for filtering single pixel spectra
    # txtfile.write('# Sav-Golay parameters\n')
    # txtfile.write('sav_gol_width = '+str(info_dict['sav_gol_width'])+'\n'+'sav_gol_poly_order = '+str(info_dict['sav_gol_poly_order'])+'\n\n')
    
    # write SNIP parameters
    txtfile.write('# SNIP parameters\n')
    txtfile.write('snip_width = '+str(info_dict['snip_width'])+'\n'+'snip_iter = '+str(info_dict['snip_iter'])+'\n\n')   
    
    # write fit parameters
    txtfile.write('# sigma bound parameters\n')
    txtfile.write('sigma_ub = '+str(info_dict['sigma_bounds'][1])+'\n'+'sigma_lb = '+str(info_dict['sigma_bounds'][0])+'\n\n')
    
    txtfile.write('# delta_peak and nfev parameters\n')
    txtfile.write('ftol = '+str(info_dict['ftol'])+'\n'+'nfev = '+str(info_dict['nfev'])+'\n'+'OPT_method = '+str(info_dict['OPT_method']))
    
    txtfile.close()        


def plotly_xfr_sum(xrfdata,info_dict):
    
    from scipy.optimize import Bounds, minimize
    import plotly.graph_objects as go
    import time

    start_time = time.time()

    # extrapolate info from dict
    fit_bounds = tuple(info_dict['fit_bounds'])
    xdata = np.asarray(info_dict['xdata'],float)[fit_bounds[0]:fit_bounds[1]]    
    xrfdata = xrfdata[fit_bounds[0]:fit_bounds[1]]            
    
    # plot
    xrfdatasum = correct_decimal_spectrum(xrfdata)
    bkg_vox = correct_decimal_spectrum(snip_bkg(xrfdatasum,np.float64(info_dict['snip_width']),np.float64(info_dict['snip_iter'])))
    
    # prepare initial parameters + boundaries and calculate fit
    init_params_vox, lb_vox, ub_vox = calc_init_params_scattering(xrfdatasum,bkg_vox,info_dict['peak_centers_ch'],info_dict['sigma_bounds'],np.int64(info_dict['n_lines']),info_dict['lines_intensity'],info_dict['lin_coords_lines'],np.int64(info_dict['beam_en_ch']),np.int64(info_dict['delta_peak']))
    bounds_tmp = Bounds(lb_vox,ub_vox)
    
    line_params = minimize(minimise_model,init_params_vox,args=(xdata, xrfdatasum - bkg_vox, np.asarray(info_dict['lines_energy_scattering'],float)),method=str(info_dict['OPT_method']),options={'ftol':np.float64(info_dict['ftol']), 'maxiter':np.int64(info_dict['nfev'])},bounds=bounds_tmp)
    
    fit_params = line_params['x']
    fit = correct_decimal_spectrum(np.sum(model_func_sum_components(xdata,np.asarray(info_dict['lines_energy_scattering'],float),*fit_params),0))
    
    # plot data + fit
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xdata, y=xrfdatasum, mode='lines', showlegend=True, name='data'))
    fig.add_trace(go.Scatter(x=xdata, y=fit + bkg_vox, mode='lines', showlegend=True, name='fit'))
    fig.add_trace(go.Scatter(x=xdata, y=bkg_vox, mode='lines', showlegend=True, name='background'))
    fig.update_yaxes(type="log")
    fig.update_layout(yaxis=dict(title_text='Counts'),xaxis=dict(title_text='Energy [eV]'),font_size=18)
    
    fit_sum_exec_time = (time.time() - start_time) * 2.0
    
    return fig, fit_sum_exec_time


def plotly_sumdata_channels(xrfdata,xdata):
    
    import plotly.graph_objects as go
    
    xrfdatasum = correct_decimal_spectrum(xrfdata)
    # plot data + fit
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xdata, y=xrfdatasum, mode='lines', showlegend=True, name='data'))    
    fig.update_yaxes(type="log")
    fig.update_layout(yaxis=dict(title_text='Counts'),xaxis=dict(title_text='Energy [eV]'),font_size=18)
    
    return fig
    
    
def calc_n_lines(element_list,lines_list):
    
    n_lines = 0
    for idx_element in range(len(element_list)):
        for idx_fam in range(len(lines_list[idx_element])):
            for idx_line in range(len(lines_list[idx_element][idx_fam])):
                n_lines += 1
                
    return n_lines


def calculate_fit_params(element_list,lines_list,lines_energy,beam_en,xdata):
    
    # initialisations
    n_lines = calc_n_lines(element_list,lines_list)
    peak_centers_ch = np.zeros((n_lines,),int)
    beam_en_ch = np.argmin(np.abs(xdata - beam_en))
    
    idx = 0
    for idx_element in range(len(lines_list)):
        for idx_fam in range(len(lines_list[idx_element])):            
            for idx_line in range(len(lines_list[idx_element][idx_fam])):
                peak_centers_ch[idx] = np.argmin(np.abs(xdata - lines_energy[idx]))
                idx += 1
                
    return n_lines, peak_centers_ch, beam_en_ch


def merge_chunks_flat(chunks_list,n_pixels):
    
    # initialisations
    n_params = np.shape(chunks_list[0][0])[1]
    chunk_size = np.shape(chunks_list[0][0])[0]
    fit_params = np.zeros((n_pixels,n_params),float)

    for idx_obj in range(len(chunks_list)):
        idx_chunk = chunks_list[idx_obj][1]
        fit_params[idx_chunk*chunk_size:(idx_chunk + 1)*chunk_size,:] = chunks_list[idx_obj][0]
            
    return fit_params


def merge_chunks_flat_remainder(chunks_list,n_pixels):
    
    # initialisations
    n_params = np.shape(chunks_list[0][0])[1]
    fit_params = np.zeros((n_pixels,n_params),float)

    for idx_obj in range(len(chunks_list)):
        idx_chunk = chunks_list[idx_obj][1]
        
        if idx_chunk == len(chunks_list) - 1:
            fit_params = chunks_list[idx_obj][0]
            
    return fit_params


def fit_wrapper_scattering(xrfdata,info_dict):
    
    import multiprocessing as mp
    import time
    
    # initialisations
    n_workers = mp.cpu_count()
    chunk_size, chunk_remainder = divmod(np.shape(xrfdata)[0],n_workers)
    fit_params_flat = np.zeros((np.shape(xrfdata)[0],(info_dict['n_lines']+1)*2,info_dict['n_SDDs']))
    fit_bounds = info_dict['fit_bounds']
    xdata = info_dict['xdata'][fit_bounds[0]:fit_bounds[1]].astype(np.float64)
    
    # chunk_size = 50
    start_time = time.time()
    
    # fit chunk
    for idx_sdd in range(info_dict['n_SDDs']):
        
        # data_map_flat = data_map.flat[:10].copy()
        data_map_flat = np.zeros((n_workers,chunk_size,np.shape(xrfdata)[1]))
        idx_coord = 0
        for idx_worker in range(n_workers):
            for idx_chunk in range(chunk_size):
                data_map_flat[idx_worker,idx_chunk,:] = xrfdata[idx_coord,:,idx_sdd]
                idx_coord += 1
                
        jobs = []        
                
        p = mp.Pool(n_workers)
        
        for idx_worker in range(n_workers):
            p.apply_async(fit_chunk_scattering,(data_map_flat[idx_worker],info_dict,idx_worker), callback=jobs.append)
           
        p.close()
        p.join()
        
        fit_params_flat[:,:,idx_sdd] = merge_chunks_flat(jobs,np.shape(xrfdata)[0])
        
        
    # fit remainder    
    for idx_sdd in range(info_dict['n_SDDs']):
        job = fit_chunk_scattering(xrfdata[-chunk_remainder:,:,idx_sdd],info_dict,idx_worker)
        fit_params_flat[-chunk_remainder:,:,idx_sdd] = job[0]
        
    print("fitting results with mp in --- %s seconds ---" % (time.time() - start_time))
    
    auc_map = fitparams2auc_flat(fit_params_flat,info_dict['lines_energy_scattering'],xdata,2)
    
    fits = FitParams2FitCurves(xdata,info_dict['lines_energy_scattering'],fit_params_flat)
    
    return fit_params_flat, auc_map, fits


def FitParams2FitCurves(xdata,mu,fit_params_flat):
    
    # initialisations
    fit_curves = np.zeros((np.shape(fit_params_flat)[0],np.shape(xdata)[0],np.shape(fit_params_flat)[2]),np.float64)
    
    for idx_sdd in range(np.shape(fit_params_flat)[2]):
        for idx_coord in range(np.shape(fit_params_flat)[0]):
            fit_curves[idx_coord,:,idx_sdd] = np.sum(model_func_sum_components(xdata,mu,*fit_params_flat[idx_coord,:,idx_sdd]),0)
        
    return fit_curves 


def FitParams2FitCurves_SDDsum(xdata,mu,fit_params_flat):
    
    # initialisations
    fit_curves = np.zeros((np.shape(fit_params_flat)[0],np.shape(xdata)[0]))
    
    for idx_coord in range(np.shape(fit_params_flat)[0]):
        fit_curves[idx_coord,:] = np.sum(model_func_sum_components(xdata,mu,*fit_params_flat[idx_coord,:]),0)
        
    return fit_curves 


def fit_wrapper_scattering_SDDsum(xrfdata,info_dict):
    
    import multiprocessing as mp
    import time
    
    @njit
    def chunk_xrfdata(xrfdatasum,n_workers,chunk_size):
        
        # data_map_flat = data_map.flat[:10].copy()
        data_map_flat = np.zeros((n_workers,chunk_size,np.shape(xrfdatasum)[1]),numba.float64)
        idx_coord = 0
        for idx_worker in range(n_workers):
            for idx_chunk in range(chunk_size):
                data_map_flat[idx_worker,idx_chunk,:] = xrfdatasum[idx_coord,:]
                idx_coord += 1
                
        return data_map_flat   
    
    # initialisations
    n_workers = mp.cpu_count()
    chunk_size, chunk_remainder = divmod(np.shape(xrfdata)[0],n_workers)
    fit_params_flat = np.zeros((np.shape(xrfdata)[0],(info_dict['n_lines']+1)),float)
    
    # chunk_size = 50
    start_time = time.time()
    
    # prepare chunks
    data_map_flat = chunk_xrfdata(xrfdata,n_workers,chunk_size)
                
    # fit chunks in parallel
    jobs = []             
    p = mp.Pool(n_workers)
    for idx_worker in range(n_workers):
        p.apply_async(fit_chunk_scattering,(data_map_flat[idx_worker],info_dict,idx_worker), callback=jobs.append)
       
    p.close()
    p.join()
    
    # merge fitted chunks back together
    fit_params_flat = merge_chunks_flat(jobs,np.shape(xrfdata)[0])
        
    # fit remainder pixels in parallel
    if chunk_remainder > 0:        
        fit_params_flat[-chunk_remainder:,:], _ = fit_chunk_scattering(xrfdata[-chunk_remainder:,:],info_dict,idx_worker)
        
    print("fitting results with mp in --- %s seconds ---" % (time.time() - start_time))
    
    fits = FitParams2FitCurves_SDDsum(np.asarray(info_dict['xdata'],float),np.asarray(info_dict['lines_energy_scattering'],float),fit_params_flat)
        
    return fit_params_flat, fits
    
    
def gen_PTE_list():
    
    import xraydb
    
    PTE = []
    for idx_el in range(1,99,1):
        PTE.append({'label':xraydb.atomic_symbol(idx_el), 'value':xraydb.atomic_symbol(idx_el)})
        
    return PTE


def gen_PTE_indexing():
    
    import xraydb
    
    PTE = dict()
    for idx_el in range(1,99,1):
        PTE.update({str(xraydb.atomic_symbol(idx_el)):idx_el})
        
    return PTE


def reg_temperature_plotly(cmap_file,cmap_name,pl_entries):

    import pandas as pd    
    import os

    cmap_path = os.path.join(os.getcwd(),cmap_file[1:])
    h = 1.0/(pl_entries-1)
    
    df = pd.read_csv(cmap_path,sep=' ')*pl_entries
    df.columns = ['R','G','B']
    
    cmap_list = []
    
    for idx_line in range(pl_entries):
        cmap_list.append([h*idx_line, 'rgb' + str((np.int64(df['R'][idx_line]),np.int64(df['G'][idx_line]),np.int64(df['B'][idx_line])))]) 
    
    return cmap_list


def fill_auc_mask(auc_map_flat,mask_idxs,im_shape):
    
    # initialisations
    auc_map = np.zeros((im_shape[0],im_shape[1],np.shape(auc_map_flat)[1]),dtype=np.int64)
        
    for idx_flat in range(len(mask_idxs)):
        coord =  np.unravel_index(mask_idxs[idx_flat],(im_shape[0],im_shape[1]))
        auc_map[coord[0],coord[1],:] = auc_map_flat[idx_flat,:].copy()
        
    return auc_map