#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:46:25 2018

@author: amit

==================================================================================
Compute LCMV/DICS beamformer (Volumetric) for all phantom data (moving or static)
==================================================================================

This is the same as /net/bonsai/home/amit/Dropbox/Python/EN_master_scripts/EN_BF_Phantom_General_for_All_Sites.py 
with the function EN_set_directory.py to detect data directory
 
"""
# Import dependencies
import matplotlib.pyplot as plt
#plt.rcParams['axes.facecolor']='white'
import numpy as np
from scipy.io import loadmat, savemat
import os
from os.path import exists
import sklearn as sk
import mne
#from mne.beamformer import lcmv, dics
from datetime import datetime
#from collections import OrderedDict
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from scipy.spatial import ConvexHull#, Delaunay
from itertools import product
#from mayavi import mlab
plt.close('all')
print(__doc__)
mne.set_log_level('WARNING')
#import warnings
#warnings.filterwarnings('ignore')
#strt = 0

def define_source_space_and_plot(mode, plot_alignment, gridres=5.0, spacing='oct6', surf_type='pial', mindist=2.5, exclude=10.0): # Compute and plot Source Space 
    if mode=='surface':  # Cortically constrained source space
            src_s='%s%s/%s-surface-%s-%s_src.fif'%(subjects_dir, subject, subject, spacing, surf_type)
            if not exists(src_s):
                src_s = mne.setup_source_space(subject=subject, spacing=spacing, surface='pial',
                                        subjects_dir=subjects_dir, add_dist=True, n_jobs=1, verbose=True)
                mne.write_source_spaces('%s%s/%s-surface-%s-%s_src.fif'%(subjects_dir, subject, subject, spacing, surf_type), src_s, overwrite=True)
            else:
                src_s=mne.read_source_spaces(src_s, patch_stats=False, verbose=True)
            print(src_s)
            src_s = '%s%s/%s-surface-%s-%s_src.fif'%(subjects_dir, subject, subject, spacing, surf_type)
            src = src_s
            del src_s
    if not mode=='surface': # Volumetric source space
        src_v = '%s%s/%s-grid%.1fmm-ex%.1fmm-md%.1fmm-vol_src.fif'%(subjects_dir, subject, subject, gridres, exclude, mindist)
        if not exists(src_v):
            src_v = mne.setup_volume_source_space(subject=None, pos=gridres, mri=mrifile,
                        bem=None, surface=surffile, mindist=mindist, exclude=exclude, 
                        subjects_dir=None, volume_label=None, add_interpolator=True, verbose=True)
            mne.write_source_spaces('%s%s/%s-grid%.1fmm-ex%.1fmm-md%.1fmm-vol_src.fif'%(subjects_dir, subject, subject, gridres, exclude, mindist), src_v, overwrite=True)
        else:
            src_v = mne.read_source_spaces(src_v, patch_stats=False, verbose=True)
        print(src_v)
        src_v = '%s%s/%s-grid%.1fmm-ex%.1fmm-md%.1fmm-vol_src.fif'%(subjects_dir, subject, subject, gridres, exclude, mindist)
        src = src_v
        del src_v 
    if plot_alignment=='yes':
        mne.viz.plot_alignment(info, trans, subject=subject, subjects_dir=subjects_dir, fig=None,
                               surfaces=['head-dense', 'inner_skull'], coord_frame='head', show_axes=True,
                               meg=False, eeg='original', dig=True, ecog=True, bem=None, seeg=True,
                               src=mne.read_source_spaces(src), mri_fiducials=False,  verbose=True) 
    return src

#%% Set data category, site etc
for site in ['Aston', 'Aston_mov', 'Aalto', 'Bari', 'Aalto_mov', 'Biomag',]: # '',
    if site=='Aalto':
        maxf, amps, dipoles = ['', '_sss'], ['20', '100','200', '1000'], (5,6,7,8) #
    if site== 'Aston':
        maxf, amps, dipoles = ['','_sss'], ['20','200','1000'], (5,6,7,8,9,10,11,12) #
    if site== 'Bari':
        maxf, amps, dipoles = ['', '_sss'], ['25'], np.arange(5,13)
    if site== 'Biomag':
        maxf, amps, dipoles = ['', '_sss'], ['500'], np.arange(5,13) # 
    if site== 'Aston_mov':
        maxf, amps, dipoles, phantom_moving = ['','_tsss_mc'], ['200'], (5,6,7,8,9,10,11,12), 'yes'
    if site== 'Aalto_mov':
        maxf, amps, dipoles, phantom_moving = ['','_tsss_mc'], [''], (1,2,3), 'yes' # Aalto_mov dipole numbers are rotation
                                                                                    #  type to fit in the same script
    datacat = 'phantom'
    apply_lcmv='yes'
    apply_dics='no'
    reg_compare='no'
    dics_freq=[5.0, 35.0]
    chancat=[True]#, 'grad', 'mag']
    print(maxf, amps, dipoles)
    ###reg_form='snr_dict[dfname][0]**4/500'
    reg_form='SNR**4/500'
    comments = reg_form + ' rank cut not used (None)'
    transit = 'par["trialwin"][1]*0.1' # 10%
    cov_meths = ['empirical']#'ledoit_wolf', 'oas']#,'diagonal_fixed' , 'shrunk',,'shrinkage' ]
    weight_norms = ['nai']#, 'unit-noise-gain']
    
#%% Run the loop
    for weight_norm, cov_meth, meg in product(weight_norms, cov_meths,chancat):
        for prepcat in maxf:
            for subcat in amps:
                for dipnum in dipoles:#dipoles:
#                    strt=strt+1
                    print(site, meg, prepcat, subcat, dipnum, cov_meth, weight_norm)
                    #%% Set parameters dictionary
                    par={'do_dipfit'  : '',
                        'other_SL'    : 'no', 
                        'savefig'     : '',
                        'save_resplot': '',
                        'maxfilter'   : '',
                        'rawfilter'   : 'yes',
                        'epochsfilter': 'no',
                        'bandpass'    : 'yes',
                        'bpfreq'      : [1, 45],
                        'calc_src_v'  : '',
                        'gridres'     : 5.0,
                        'mindist'     : 5.0,
                        'exclude'     : 0.0,
                        'bem_sol'     : 'sphere',
                        'check_trial' : '',
                        'browse'      : '',
                        'more_plots'  : '',
                        'models_plot' : '',                
                        'numtry'      : 2}
                        
                    par['event_dict']= dict(Dipole=dipnum)
                    par['event_id']= dipnum
                                
                    # Load actual dipole location for phantom
                    if site in {'Bari', 'Aston', 'Aston_mov', 'JYU', 'Biomag', 'Birmingham'}:
                        act_dip=loadmat('/net/bonsai/home/amit/Documents/MATLAB/biomag_phantom.mat') # TRIUX
                        act_dip=act_dip['biomag_phantom']
                    elif site in {'Aalto', 'Aalto_mov','Biomag_old'}:
                        act_dip=loadmat('/net/bonsai/home/amit/Documents/MATLAB/aalto_phantom.mat')  # Vectorview
                        act_dip=act_dip['aalto_phantom']
                    
                    # Phantom CT information
                    subjects_dir, subject = '/opt/freesurfer/subjects/', 'Aalto_phantom'
                    trans = subjects_dir + subject + '/mri/transforms/' + 'Aalto_phantom-trans2.fif'
                    mrifile= subjects_dir + subject + '/mri/T1.mgz'
                    surffile= subjects_dir + subject + '/bem/watershed/' + subject + '_outer_skin_surface'
                    runfile('/net/bonsai/home/amit/Dropbox/Python/EN_master_scripts/EN_set_directory.py')
        #%% Set MRI/MEG data and data specific constarints
                    [data_path, fname, out_dir] = EN_set_directory(site, datacat, subcat, dipnum, prepcat)
                    dfname=os.path.split(os.path.splitext(fname)[0])[1]
                    out_path = out_dir + 'MNE//'
        
                    if site=='Aalto': # Aalto old data shared very first time 
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
                        badch = raw.info['bads'] = []
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG2233', 'MEG2231', 'MEG0111', 'MEG2422', 'MEG1842', 'MEG0511']  
                        par['stimch']  = 'STI201'
                        par['trialwin'] = [-0.100, 0.100]
                        par['ctrlwin']  = [-0.100, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.100]
                        #par['bpfreq']  = [1, 40]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [], ''
                        dfname=os.path.split(os.path.splitext(fname)[0])[1]
                        events = mne.find_events(raw, stim_channel=par['stimch']) 
                        #np.save(out_dir + '/Aalto_phantom_SNR.npy', snr_dict)
                        #snr_dict=np.load(out_dir + '/Aalto_phantom_SNR.npy').item()

                    if site=='Aston': # Aston shared static & moving phantom data  
                        mov=''
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
                        badch = raw.info['bads'] = []
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG1133', 'MEG1323', 'MEG0613', 'MEG1032'] # added 'MEG0613', 'MEG1032' 
                        par['stimch']  = 'SYS201'
                        par['trialwin'] = [-0.5, 0.5]
                        par['ctrlwin']  = [-0.5, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.5]
                        #par['bpfreq']   = [2, 95]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [0], 'resetsignal'
                        dfname=os.path.split(os.path.splitext(fname)[0])[1]
                        events = mne.find_events(raw, stim_channel=par['stimch']) # Read even
                        
                    if site=='Biomag': # Aston shared static & moving phantom data  
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
                        badch = raw.info['bads'] = []
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG0342',  'MEG0542', 'MEG1013'] # added 'MEG0613', 'MEG1032' 
                        par['stimch']  = 'SYS201'
                        par['trialwin'] = [-0.1, 0.1]
                        par['ctrlwin']  = [-0.1, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.1]
                        #par['bpfreq']   = [2, 95]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [0], 'resetsignal'
                        dfname=os.path.split(os.path.splitext(fname)[0])[1]
                        events = mne.find_events(raw, stim_channel=par['stimch']) # Read even
                        
                    if site=='Bari': # Bari static phantom data  
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=True, preload=True, verbose=True)
                        par['stimch']  = 'SYS201'
                        dfname=os.path.split(os.path.splitext(fname)[0])[1] #+ '-dip'+str(dipnum)
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG0943', 'MEG0222', 'MEG1522', 'MEG1512', 'MEG1432', 'MEG1113', 'MEG0631']
                        par['trialwin'] = [-0.1, 0.1]
                        par['ctrlwin']  = [-0.1, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.1]
                        #par['bpfreq']   = [2, 95]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [], 'resetsignal'
#                        croptimes=np.load('/net/bonsai/home/amit/Dropbox/Python/Bari_phantom_All_dipole_25nAmp1_sss_croptime.npy')
#                        raw.crop(croptimes[dipnum,:][0], croptimes[dipnum,:][1])
                        events = mne.find_events(raw, stim_channel=par['stimch']) # Read event
#                        events=np.concatenate([np.transpose([events[:,0]-events[0,0]]), events[:,1:3]], 1)
                        
#                        kk=events[np.where(events==dipnum)[0],:]
#                        croptime=[kk[0,0]/raw.info['sfreq'],kk[-1,0]/raw.info['sfreq']]
#                        raw.crop(tmin=croptime[0],tmax=croptime[-1])
#                        events=np.concatenate([np.transpose([kk[:,0]-kk[0,0]]), kk[:,1:3]], 1)
#                        events=events[1:-2, :]
                        
                    if site=='Aston_mov': # Aston shared static & moving phantom data  
                        mov='_movement'
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
                        badch = raw.info['bads'] = []
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG1133', 'MEG1323', 'MEG0613', 'MEG1032'] # added 'MEG0613', 'MEG1032' 
                        par['stimch']  = 'SYS201'
                        par['trialwin'] = [-0.5, 0.5]
                        par['ctrlwin']  = [-0.5, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.5]
                        #par['bpfreq']   = [2, 95]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [0], 'resetsignal'
                        dfname=os.path.split(os.path.splitext(fname)[0])[1]
                        events = mne.find_events(raw, stim_channel=par['stimch']) # Read even
                        events[:,2]=events[:,2]-events[:,1]
                        events[:,1]=events[:,1]-events[:,1]
                        
                    if site=='Aalto_mov': # Aston shared static & moving phantom data  
                        mov='_movement'
                        raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
                        badch = raw.info['bads'] = []
                        if prepcat=='':
                            badch = raw.info['bads'] = ['MEG1522'] # added 'MEG0613', 'MEG1032' 
                        par['stimch']  = 'STI201'
                        par['trialwin'] = [-0.1, 0.1]
                        par['ctrlwin']  = [-0.1, -eval(transit)]
                        par['actiwin']  = [eval(transit), 0.1]
                        #par['bpfreq']   = [2, 95]
                        reject=dict(grad=4000e-13, mag=4e-12)
                        badtrial, badreason = [0], 'resetsignal'
                        dfname=os.path.split(os.path.splitext(fname)[0])[1]
                        events = mne.find_events(raw, stim_channel=par['stimch']) # Read even
                        events[:,2]=events[:,2]-events[:,1]
                        events[:,1]=events[:,1]-events[:,1]
                        dipnum= 2
                        par['event_dict']= dict(Dipole=dipnum)
                        par['event_id']= dipnum
                    
#                    snr_dict=np.load(out_dir + site + '_' + datacat + '_SNR_' + str(meg) + '.npy').item()
                    #raw.drop_channels(raw.info['bads'])
                    raw.pick_types(meg=meg, exclude='bads')
                    raw.info
                    
                    out_dir = '/net/qnap/data/rd/ChildBrain/BeamComp/Result_master/' + site + '_phantom_data/'
                    out_path = out_dir + 'MNE/'                   
    
                    if not os.path.exists(out_path):
                            os.mkdir(out_path)
                    #resultfile = out_path + 'MNEp_Result-numtry' + str(par['numtry'])+ '-phantom-'+ site + '_source_loc.csv'
                    resultfile = '/net/bonsai/home/amit/Dropbox/BeamComp_Resultfiles/' + site + '_phantom_data/' 'MNE/'+ 'MNEp_Result-numtry' + str(par['numtry'])+ '-phantom-'+ site + '_source_loc.csv'
                    resultfile_dics = resultfile[:-4]+'-Reg_Compare.csv'
                    if meg==chancat[0] and subcat==amps[0] and prepcat==maxf[0] and dipnum==dipoles[0]:
                        fid = open(resultfile, 'a+')
                        fid.writelines('=======================================================================================\n'+
                                       'raw.info["bads"] = %s'%raw.info['bads'] + '(only for raw data not for MaxFiltered)\n' +
                                       'par["trialwin"]= %s'%par['trialwin']+ '\n' +
                                       'par["ctrlwin"]= %s'%par['ctrlwin']+ '\n' +
                                       'par["actiwin"]= %s'%par['actiwin']+ '\n' +
                                       'par["bpfreq"]= %s'%par['bpfreq']+ '\n' +
                                       'par["gridres"]= %s'%par['gridres']+ '\n' +
                                       'par["mindist"]= %s'%par['mindist']+ '\n' +
                                       'par["exclude"]= %s'%par['exclude']+ '\n' +
                                       'par["bem_sol"]= %s'%par['bem_sol']+ '\n' +
                                       'Date & Time = %s' %str(datetime.now())+ '\n' +
                                       'Regularization method= %s'%reg_form +'\n' +
                                       '=======================================================================================\n')
                        fid.close()
                        if apply_dics=='yes':
                            fid = open(resultfile_dics, 'a+')
                            fid.writelines('=======================================================================================\n'+
                                           'raw.info["bads"] = %s'%raw.info['bads'] + '(only for raw data not for MaxFiltered)\n' +
                                           'par["trialwin"]= %s'%par['trialwin']+ '\n' +
                                           'par["ctrlwin"]= %s'%par['ctrlwin']+ '\n' +
                                           'par["actiwin"]= %s'%par['actiwin']+ '\n' +
                                           'par["bpfreq"]= %s'%par['bpfreq']+ '\n' +
                                           'par["gridres"]= %s'%par['gridres']+ '\n' +
                                           'par["mindist"]= %s'%par['mindist']+ '\n' +
                                           'par["exclude"]= %s'%par['exclude']+ '\n' +
                                           'par["bem_sol"]= %s'%par['bem_sol']+ '\n' +
                                           'Date & Time = %s' %str(datetime.now())+ '\n' +
                                           'Regularization method= %s'%reg_form +'\n' +
                                           '=======================================================================================\n')
                            fid.close()
                            
                            # Save data for ploting againt other package data >>>>>>>>>>>>>>>>>>>>
                            matdata = {}
                            matdata['data']=raw.get_data()
                            matdata['label']=raw.ch_names
                            savemat('%sMNE_data_%s.mat'%(data_path,dfname), matdata)
                            del matdata
                        
        #%% Browse data
                    if par['browse']=='yes':
                        mne.viz.plot_events(events, sfreq=None, first_samp=0, color=None, event_id=par['event_dict'], 
                            axes=None, equal_spacing=True, show=True)
                        raw.plot(events=events, duration=1.0, start=0.0, n_channels=20, bgcolor='w', color=None,
                                 bad_color=(0.5, 0.1, 0.1), event_color='cyan', scalings=None, remove_dc=True,
                                 order='type', show_options=False, title=fname, show=True, block=False,
                                 highpass=None, lowpass=None, filtorder=4, clipping=None, show_first_samp=False)
                        raw.plot_psd(tmin=0.0, tmax=raw.times[-1], fmin=0, fmax=300, proj=False, n_fft=None, picks=None, ax=None, 
                                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
        
        #%% Apply filter if required
                    if par['maxfilter']=='yes'and prepcat=='':
                        raw.fix_mag_coil_types()
                        raw=mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.0), int_order=8, ext_order=3, 
                                calibration=None, cross_talk=None, st_duration=None, st_correlation=0.98, coord_frame='head',
                                destination=None, regularize='in', ignore_ref=False, bad_condition='error', head_pos=None,
                                st_fixed=True, st_only=False, mag_scale=100.0, verbose=None)
                        if par['more_plots']=='yes':
                            raw.plot(events=events, title='Raw > MNE SSSed data plot')
                            raw.plot_psd(tmin=0.0, tmax=raw.times[-1], fmin=0, fmax=300, proj=False, n_fft=None, picks=None, ax=None, 
                                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
                            
                    if par['rawfilter']=='yes' and par['epochsfilter']!='yes':
                        if par['bpfreq'][1]>48:
                            raw.notch_filter(50, filter_length='auto', phase='zero') # Notch filter
#                        raw.filter(par['bpfreq'][0], par['bpfreq'][1], l_trans_bandwidth=0.1,#min(max(2 * 0.01, 2), 2), 
#                                   h_trans_bandwidth=min(max(70 * 0.01, 2.), raw.info['sfreq'] / 2. - 70), 
#                                   filter_length='auto', phase='zero')
                        raw.filter(par['bpfreq'][0], par['bpfreq'][1], filter_length='auto', phase='zero')
                        if par['more_plots']=='yes':
                            raw.plot(events=events, title='Raw> MNE SSSed data> Notch & badpassed data plot')
                            raw.plot_psd(tmin=0.0, tmax=raw.times[-1], fmin=par['bpfreq'][0], fmax=par['bpfreq'][1], proj=False, n_fft=None, picks=None, ax=None, 
                                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
                    raw.plot_psd(average=False, spatial_colors=True, line_alpha=0.5, fmin=0.0, fmax=100.0) 
                    plt.savefig(out_path + dfname + '_raw_plot_psd.png', facecolor='w', edgecolor='w', 
                                        orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                    matdata = {}
                    matdata['data']=raw.get_data()
                    matdata['label']=raw.ch_names
                    savemat('%sMNE_data_%s_filtered.mat'%(data_path,dfname), matdata)
                    del matdata
        
        #%% Pick channels              
                    # raw.info.normalize_proj()             
                    epochs = mne.Epochs(raw, events, par['event_id'], par['trialwin'][0], 
                                        par['trialwin'][1], baseline=tuple(par['ctrlwin']), picks=None, 
                                        preload=True, reject=None, flat=None, proj=False, decim=1,
                                        reject_tmin=None, reject_tmax=None, detrend= None, 
                                        on_missing='error', reject_by_annotation=True, verbose=True)
                    epochs.drop(badtrial, badreason)
                    if site=='Aston' and prepcat=='_sss' and subcat=='200' and dipnum==12:
                        epochs.drop([37,38,39]) 
#                    elif site=='Bari':
#                        epochs.drop(np.arange(0,13))
                    del raw
                    if par['epochsfilter']=='yes' and par['rawfilter']!='yes':
                        epochs.filter(par['bpfreq'][0], par['bpfreq'][1], picks=None, filter_length=100, l_trans_bandwidth=2, 
                                      h_trans_bandwidth=2, n_jobs=1, method='fir', iir_params=None, 
                                      phase='zero', fir_window='hamming', fir_design='firwin2', pad='edge', 
                                      verbose=True)
                        epochs.filter(51, 49, picks=None, filter_length=100, l_trans_bandwidth=2, 
                                      h_trans_bandwidth=2, n_jobs=1, method='fir', iir_params=None, 
                                      phase='zero', fir_window='hamming', fir_design='firwin2', pad='edge', 
                                      verbose=True)
                    
                    if par['check_trial']=='yes':
                        epochs.plot(picks=None, scalings=None, n_epochs=10, n_channels=30, event_colors=None, 
                                    title='Epochs plot (cascaded)', events=None, show=True, block=False)
                        
                    # epochs.plot_drop_log() 
                    epochs.save(fname[0:-4] + '_%s-%sHz_mne_epo.fif' %tuple(par['bpfreq'])) #(data_path + dfname + '-epo.fif')
                    # epochs=mne.read_epochs(data_path + dfname + '-epo.fif', proj=True, preload=True, verbose=True)
                    info=epochs.info
        #%% Average & plot            
                    evoked= epochs.average() # Average epochs
                    if not os.path.exists(fname[0:-4] + '_%s-%sHz_mne_ave.fif' %tuple(par['bpfreq'])):
                        mne.write_evokeds(fname[0:-4] + '_%s-%sHz_mne_ave.fif' %tuple(par['bpfreq']), evoked)  # for Xfit test
#                    evoked.plot(spatial_colors=True, gfp=True)

        #%%Compute Source Space .................................>>>>
                    src = define_source_space_and_plot(mode='vol', plot_alignment='yes', gridres=5.0, spacing='ico4', surf_type=None, mindist=5.0, exclude=10.0)
        #%% Model BEM
                    bem=mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=None, info=None, verbose=True)

        #%% Forward solution..................>> 
                    if not 'fwd' in locals():
                        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                                                    meg=meg, eeg=False, mindist=None, n_jobs=1)
                        fwd['site'] = site
                    if not fwd['site']==site:
                        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                                                    meg=meg, eeg=False, mindist=None, n_jobs=1)
                        fwd['site'] = site                        
                    nchan_fwd = fwd['sol']['data'].shape[0]
                    if len(epochs.ch_names)!=nchan_fwd:
                        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                                                    meg=meg, eeg=False, mindist=None, n_jobs=1)
                        fwd['site'] = site
                    print("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)
                    # plt.figure(), plt.imshow(np.dot(fwd['sol']['data'], fwd['sol']['data'].T))
                    #resultfile = out_path + 'MNEp_Result-numtry' + str(par['numtry'])+ '-phantom-'+ site + '_source_loc.csv'
                    
                    #%%
#                    evoked_pre = evoked.copy().crop(tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1])
#                    evoked_pst = evoked.copy().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
#                                        
#                    varpst = np.var(evoked_pst.data, axis=1)
#                    
#                    ch_idx = np.where(varpst>=max(varpst)*0.75)[0]
#                    
#                    ch_name = [evoked_pst.ch_names[i] for i in ch_idx ]
#                    
#                    SNR = snr(avgpst.avg(ch_idx,:)', avgpre.avg(ch_idx,:)');
#                
#                    par.reg = eval(reg_form);
        #%% Apply time domain Beamforming (LCMV):
                    if apply_lcmv=='yes':
                        #cov_meth ='shrinkage'
                        evoked_pst = evoked.copy().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
                        evoked_pre = evoked.copy().crop(tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1])
#                        evoked_pst.plot(spatial_colors=True, gfp=True, time_unit='ms')
#                        evoked_pre.plot(spatial_colors=True, gfp=True, time_unit='ms')
#                        evoked = evoked_pst
                        noise_cov = mne.compute_covariance(epochs, tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1], method=cov_meth) 
                        data_cov = mne.compute_covariance(epochs, tmin=par['actiwin'][0], tmax=par['actiwin'][1], method=cov_meth)   
                        noise_cov.save('%s_%s_noise_cov.fif'%(fname[:-4],cov_meth))
                        data_cov.save('%s_%s_data_cov.fif'%(fname[:-4],cov_meth))
#           
                        inverse_operator=mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=1, depth=0.199, verbose=True)
                        snr, snr_est = mne.minimum_norm.estimate_snr(evoked_pst, inverse_operator, verbose=True)
                        if par['more_plots']=='yes':
                            plt.figure(dfname + '_snr&snr_est') 
                            plt.plot(snr, 'r', label='snr'),  plt.hold(True), plt.plot(snr_est, 'g', label='snr_est')
                            plt.legend(loc='upper right', fontsize=None, shadow=False, framealpha=0.3)
                            plt.suptitle(dfname + '_snr&snr_est') 
                            # plt.savefig(out_path + dfname + '_snr&snr_est', facecolor='w', edgecolor='w', orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                        peak_ch, peak_time = evoked_pst.get_peak(ch_type='mag')
                        tp = int((peak_time - par['actiwin'][0])*evoked_pst.info['sfreq'])
                        SNR=snr[tp]
                        # snr_est_mne = snr_est[tp]
                        #mne.viz.plot_snr_estimate(evoked, inverse_operator, show=True)
                        #plt.ylim(0,30)
                        SNRs=[SNR]
                        cnttt=0
                        for SNR in SNRs:
                            cnttt=cnttt+1
                            reg=eval(reg_form)
                            reg= 0.05
                            print(SNR, reg)
                            
                            #noise_cov.plot(epochs.info, show_svd=True, proj=True)
                            #data_cov.plot(epochs.info, show_svd=True, proj=True)
                            
    #                        stc = lcmv(evoked, fwd, noise_cov=noise_cov, data_cov=data_cov, reg=reg,
    #                                   pick_ori='max-power', max_ori_out='abs', reduce_rank=True, verbose=True)
    #                    kk, cnt =np.zeros((20,2)), -1
    #                    for reg in [0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
    #                        cnt = cnt + 1
#                            rank=None#np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)-1 #np.linalg.matrix_rank(evoked.copy().crop(par['ctrlwin'][0], par['ctrlwin'][1]).data), 
#                            # plt.figure('svd plot_' + dfname), plt.plot(np.linalg.svd(noise_cov.data)[1])
                            rank = None
                            #weight_norm='nai'#'unit-noise-gain'
                            if 'filters' in locals():
                                del filters
                            reducerank = 0    
                            while not 'filters' in locals():
                                try:
                                    filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=reg, 
                                                  noise_cov=noise_cov, pick_ori='max-power', rank=rank,
                                                  weight_norm=weight_norm, reduce_rank=True, verbose=True)
                                except ValueError as Val_err:
                                    print('ValueError: Reduce rank manually > > > >\n\n')
                                    reducerank = reducerank + 1
                                    rank = np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)- reducerank
                                    print('Rank reduced to > ' + str(rank))
                                except np.linalg.LinAlgError as Lin_alg_err:
                                    print('LinAlgError: Reduce rank manually > > > >\n\n')
                                    reducerank = reducerank + 1
                                    rank = np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)- reducerank
                                    print('Rank reduced to > ' + str(rank))
                                except TypeError as Type_err:
                                    print('TypeError: Increase reg. value > > > >\n\n')
                                    reg = reg*10
                                    print('Reg. value changed from %s to %s'%(str(reg/10), str(reg)))
                    
                            stc = mne.beamformer.apply_lcmv(evoked_pst, filters, max_ori_out='signed', verbose=True)
                            
                            stc=np.abs(stc)
                            src_peak, t_peak=stc.get_peak()
                            est_loc = fwd['src'][0]['rr'][src_peak]*1000 
                            timepoint = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)
                            loc_err=np.sqrt(np.sum(np.square(act_dip[dipnum-1]-est_loc))); # Calculate loc_err
            
                            print('Act_Sourceloc for %s' %dfname + '= %s' % str(act_dip[dipnum-1])) 
                            print('Est_SourceLoc for %s' %dfname + '= %s' % str(np.around(est_loc,1)))
                            print('Peak_Value for %s' %dfname + '= %.2f' % stc.data.max())
                            print('Loc_error for %s' %dfname + '= %.1f' % loc_err)
    #                        kk[cnt,0]=reg
    #                        kk[cnt,1]=loc_err
                            #% % find the focal length >>>>>>>>>>>>>>>>>>
                            stc_data = stc.copy().data
                            stc_data1 = stc_data[:,timepoint]
                            XX=stc_data1[:,np.newaxis]
                            # plt.figure(), plt.plot(XX)
                            clust_data1 = XX[XX >= (stc_data1.max() * 0.50)]*1.0e+20
                            n_act_grid = len(clust_data1)                 
                            PSVol = n_act_grid*(par['gridres']**3)               
                            # plt.figure(), plt.plot(clust_data1)
                            clust_data = XX   
                            XX[XX >= (stc_data1.max() * 0.50)] = clust_data1
                            # plt.figure(), plt.plot(XX)
                            
                            kmeans = sk.cluster.KMeans(n_clusters=2, random_state=0).fit(XX)
                            idx = kmeans.labels_
                            plt.figure(), plt.plot(idx)
                            kmeans.cluster_centers_
                            kmeans.inertia_            
                                        
                            locs1 = fwd['source_rr'][idx==1,:]
                            n_act_vert = len(locs1)
                            totalVolume2 = n_act_vert*(par['gridres']**3)
                            locs2 = fwd['source_rr'][idx==0,:]
                            cntrd_locs1 = np.array([[locs1[:,0].mean(), locs1[:,1].mean(), locs1[:,2].mean()]]);
                            loc_err2 =  np.sqrt(np.sum(np.square(act_dip[dipnum-1]-(cntrd_locs1*1000))))
                            kk=np.array([0])
                            for ii in range(n_act_grid):
                                kk = np.vstack((kk,np.sqrt(np.sum(np.square(locs1[ii,:]-cntrd_locs1)))*1000))
                            meandist = sum(kk)[0]/n_act_grid
                                
                            while len(locs1)<=4:
                                locs1 = np.vstack((locs1, locs1[0,:]))
                            hull = ConvexHull(locs1, incremental=False, qhull_options='QJ')
                            tris1 = hull.simplices
                            totalVolume = hull.volume *1000*1000*1000
                            totalArea   = hull.area *1000*1000
                            # [totalVolume3, totalArea3] = stlVolume(locs1,tris1)
                            print('Total volume = %.2fmm3 \nTotal area = %.2fmm2' %(totalVolume, totalArea))                
                            print('No. of active voxels = %d \nActual total volume = %.2fmm3' %(n_act_grid, totalVolume2))
                            if par['more_plots']=='yes':                 
                                fig = plt.figure()
                                ax = fig.add_subplot(111, projection='3d')
                                ax.scatter(locs1[:,0], locs1[:,1], locs1[:,2], c='green', marker='*', s=50)
                                ax.scatter(cntrd_locs1[:,0], cntrd_locs1[:,1], cntrd_locs1[:,2], c='orange', marker='o', s=100)
                                ax.plot_trisurf(locs1[:,0],locs1[:,1],locs1[:,2], triangles=tris1,  cmap=plt.cm.Spectral, alpha=0.8)#,
                                plt.axis('off')
                                plt.show()
                            if par['more_plots']=='yesss':
                                # plot 3D points in mayavi (FIX IT)
                                print('fix it by fixit coreg')
    #                            vrtx = range(0,len(model[0]['rr']),2)
    #                            innner_skull_head = np.dot(model[0]['rr'], trans_val[0]['trans'][:3, :3].T)
    #                            innner_skull_head2 = mne.transforms.apply_trans(trans_val[0]['trans'].T, model[0]['rr'], move=True)
    #                            mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700))
    #                            mlab.points3d(locs1[:,0], locs1[:,1], locs1[:,2],mode='sphere', scale_factor=0.005, color=(1,1,0))
    #                            mlab.points3d(cntrd_locs1[:,0], cntrd_locs1[:,1], cntrd_locs1[:,2], mode='sphere', scale_factor=0.005, color=(0.9,0.8,0.9))
    #                            mlab.points3d(innner_skull_head[vrtx,0], innner_skull_head[vrtx,1], innner_skull_head[vrtx,2], mode='sphere', scale_factor=.0005, color=(0.8,0.9,0.8))
    #                            mlab.points3d(innner_skull_head2[vrtx,0], innner_skull_head2[vrtx,1], innner_skull_head2[vrtx,2], mode='sphere', scale_factor=.0005, color=(0.1,0.9,0.8))
    #                            mlab.show()
                               
                            fid = open(resultfile, 'a+')
                            if prepcat==maxf[0] and dipnum==dipoles[0] and subcat==amps[0] and cnttt==1:
                                fid.writelines('\n%s\n'   %comments)
                                fid.writelines('%s\n%s\n%s\n' %('*********', '*********', datetime.now()))
                            fid.writelines('%s,'     %dfname)
                            fid.writelines('%s,'     %subcat)
                            fid.writelines('%.f,'    %dipnum)
                            fid.writelines('%.2f,'   %est_loc[0])
                            fid.writelines('%.2f,'   %est_loc[1])
                            fid.writelines('%.2f,'   %est_loc[2])
                            fid.writelines('%.2f,'   %stc.data.max())
                            fid.writelines('%.2f,'   %loc_err)
                            fid.writelines('%.2f,'   %np.sqrt(np.sum(np.square([0,0,0]-act_dip[dipnum-1]))))
                            fid.writelines('%.2f,'   %np.sqrt(np.sum(np.square([0,0,0]-est_loc))))
                            fid.writelines('%.f,'    %evoked_pst.nave)
                            fid.writelines('%.f,'    %len(evoked.ch_names))
                            fid.writelines('%.3f,'   %SNR)
                            fid.writelines('%.3f,'   %reg)
                            fid.writelines('%s,'     %str(rank))
                            fid.writelines('%s,'     %'')
                            fid.writelines('%s,'     %'LCMV')
                            fid.writelines('%.2f,'   %totalVolume)
                            fid.writelines('%.2f,'   %totalArea)
                            fid.writelines('%d,'     %n_act_grid)
                            fid.writelines('%.2f,'   %PSVol) 
                            fid.writelines('%.2f,'   %loc_err2)
                            fid.writelines('%.2f,'   %cntrd_locs1[0][0])
                            fid.writelines('%.2f,'   %cntrd_locs1[0][1])
                            fid.writelines('%.2f,'   %cntrd_locs1[0][2])
                            fid.writelines('%.2f,'   %meandist)
                            fid.writelines('%s,'     %cov_meth)
                            fid.writelines('%s,\n'   %weight_norm)
                            fid.close()
                            
                            if par['more_plots']=='yes':
                                plt.figure()
                                ts_show = -50  # show first 5 peak sources 
                                plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
                                plt.title(dfname + ' for %d largest sources'%abs(ts_show))
                                plt.xlabel('time (ms)')
                                plt.ylabel('%s value ' %'LCMV stc'+ '@reg=%.2f'%reg )
                                plt.show()
                                if par['save_resplot']=='yes':
                                    plt.savefig(out_path + dfname + '_STCplot.png', facecolor='w', edgecolor='w', 
                                                orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                                
                            #stc.crop(0.0, stc.times[-1])
                            if par['more_plots']=='yes':
                                thresh = stc.data.max()*70/100
                                img=mne.save_stc_as_volume('lcmv_stc.nii', stc, fwd['src'], dest='mri', mri_resolution=False)
                                plot_stat_map(index_img(img, timepoint), mrifile, threshold=thresh)
                                plt.suptitle('%s'%dfname  + ' / LCMV (tpeak=%.3f s.)' % stc.times[timepoint] + 
                                             'PeakValue= %.3f\n' % stc.data.max() + 'Reg= %.3f' % reg + 'Est_loc= %.1f' % est_loc[0] + 
                                             ', %.1f' % est_loc[1]+', %.1f ' % est_loc[2] + '/ Loc_err= %.2f mm' % loc_err, 
                                             fontsize=12, color='white')
                                if par['save_resplot']=='yes':
                                    figname = out_path + dfname +  '_SourceImage.png'
                                    plt.savefig(figname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, 
                                                        format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
                                os.remove('lcmv_stc.nii')
                            
                            plt.close('all')
        #%% Study the effect of regilarization parameter 
                        if reg_compare=='yes':
                            trial=1
                            regs=[0.001, 0.005, 0.01, 0.05,0.1,0.5,1,2,4,8,10,20,50,75,100,150]
                            if par['savefig_res']=='yes':
                                plt.ion()  # make plot interactive
                                _, ax = plt.subplots(4, 4)  # create subplots
                                plt.tight_layout()
                                ts_show = -50  # show the 200 largest responses
                                plt.suptitle('LCMV STC plot for %s-'%dfname + ' for %d largest sources '%abs(ts_show)+ 'BEM(sphere model)', fontsize=18)
                            
                            cnt=0
                            for reg in regs:
                                cnt=cnt+1
                                #stc = lcmv(evoked, fwd, noise_cov, data_cov, reg=reg, verbose=True)
                                stc = lcmv(evoked, fwd, noise_cov=noise_cov, data_cov=data_cov, reg=reg,
                                       pick_ori='max-power', max_ori_out='abs', reduce_rank=True, verbose=True)
                                stc=np.abs(stc)
                                #stc.crop(0.0, stc.times[-1])
                                v_peak, t_peak=stc.get_peak()
                                est_loc=fwd['src'][0]['rr'][v_peak]*1000 # in mili meter
                                loc_err=np.sqrt(np.sum(np.square(act_dip[dipnum-1]-est_loc))); # Calculate loc_err
                                
                                regcompresfile=resultfile[:-4]+'-Reg_Compare.csv'
                                fid = open(regcompresfile, 'a+')
                                fid.writelines('%s,' %dfname)
                                fid.writelines('%s,' %subcat)
                                fid.writelines('%.f,' %dipnum)
                                fid.writelines('%.2f,' %est_loc[0])
                                fid.writelines('%.2f,' %est_loc[1])
                                fid.writelines('%.2f,' %est_loc[2])
                                fid.writelines('%.2f,' %stc.data.max())
                                fid.writelines('%.2f,' %loc_err)
                                fid.writelines('%.f,' %len(evoked.ch_names))
                                fid.writelines('%.f,' %evoked.nave)
                                fid.writelines('%.2f,' %reg)
                                fid.writelines('%s,' %prepcat)
                                fid.writelines('%s\n' %'LCMV')
                                fid.close()
                        
                                if par['savefig_res']=='yes':
                                    ax = plt.subplot(4,4,cnt)
                                    plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
                                    #ax.set_xlabel('time (ms)')
                                    ax.set_title('EstLoc=%s ' %np.round(est_loc,2) + '/PeakVal=%.1f ' %stc.data.max() +
                                                 '/LocErr=%.1f ' % loc_err + '/@%dms'%(t_peak*1000), fontsize=11, y=0.99)
                                    ax.set_ylabel('%s ' % 'LCMV' + '@Reg= %.3f' %reg)
                                    print(str(reg) + '  done >>>>>>>>>>>>>>>')
                                    print('Loc_error for %s' %dfname + '= %.1f' % loc_err)
                                    
                            plt.pause(0.25)
                            manager=plt.get_current_fig_manager()
                            manager.window.showMaximized()
                            plt.pause(0.25)
                            plt.tight_layout()
                            plt.pause(0.25)
                            plt.subplots_adjust(top=0.94, bottom=0.01)
                            plt.pause(0.25)
                            plt.subplots_adjust(hspace=0.22, wspace=0.16)
                            figname=out_path + 'Regularization_effects_LCMV_' + dfname + str(trial) + '.png'
                            while os.path.exists(figname):
                                trial=trial+1
                                figname=out_path + 'Regularization_effects_LCMV_' + dfname + str(trial) + '.png'
                            plt.savefig(figname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, 
                                                format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
                            plt.close()
        #%% Apply frequency domain beamforming (DICS):
                    if apply_dics=='yes':
                        #reg=0.05
                        #evoked= epochs.copy().crop(par['actiwin'][0], par['actiwin'][1]).average()  
                        noise_csd = mne.time_frequency.csd_epochs(epochs, mode='multitaper', tmin=par['ctrlwin'][0], 
                                                                  tmax=par['ctrlwin'][1], fmin=dics_freq[0], fmax=dics_freq[1])
                        data_csd = mne.time_frequency.csd_epochs(epochs, mode='multitaper', tmin=par['actiwin'][0], 
                                                                 tmax=par['actiwin'][1], fmin=dics_freq[0], fmax=dics_freq[1])
                
                        stc = dics(evoked, fwd, noise_csd, data_csd, reg=reg, verbose=True)
                        
                        stc=np.abs(stc)
#                        img=mne.save_stc_as_volume(data_path + dfname  +  '_dics_stc.nii',
#                                                   stc, fwd['src'], dest='mri', mri_resolution=False)
                        
                        src_peak, t_peak=stc.get_peak()
                        est_loc = fwd['src'][0]['rr'][src_peak]*1000 
                        loc_err=np.sqrt(np.sum(np.square(act_dip[dipnum-1]-est_loc))); # Calculate loc_err
        
                        print('Act_Sourceloc for %s' %dfname + '= %s' % str(act_dip[dipnum-1])) 
                        print('Est_SourceLoc for %s' %dfname + '= %s' % str(np.around(est_loc,1)))
                        print('Peak_Value for %s' %dfname + '= %.2f' % stc.data.max())
                        print('Loc_error for %s' %dfname + '= %.1f' % loc_err)
                        
#                        plt.figure()
#                        ts_show = -50  # show first 5 peak sources 
#                        plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
#                        plt.title(dfname + ' for %d largest sources'%abs(ts_show))
#                        plt.xlabel('time (ms)')
#                        plt.ylabel('%s value ' %'DICS stc'+ '@reg=%.2f'%reg )
#                        plt.show()
                        
#                        stc.crop(0.0, stc.times[-1])
#                        thresh = stc.data.max()*40/100
#                        timepoint = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)
                        
        #                plot_stat_map(index_img(img,  int(timepoint)), 
        #                              mrifile, draw_cross=True, threshold=thresh, 
        #                              title = dfname +  '/ DICS/ '+ 'reg=%.2f'%reg + '/ Vpeak=%.3f\n'%stc.data.max()+ 
        #                              'Tpeak=%.3fs'%t_peak + '/ Est_loc= [%.1f, %.1f, %.1f]' %tuple(est_loc) + 
        #                              '/ Loc_err= %.1f' % loc_err)
                        resultfile_dics=resultfile[:-4] + '_DICS.csv'
                        fid = open(resultfile_dics, 'a+')
                        fid.writelines('%s,' %dfname)
                        fid.writelines('%s,' %subcat)
                        fid.writelines('%.f,' %dipnum)
                        fid.writelines('%.2f,' %est_loc[0])
                        fid.writelines('%.2f,' %est_loc[1])
                        fid.writelines('%.2f,' %est_loc[2])
                        fid.writelines('%.2f,' %stc.data.max())
                        fid.writelines('%.2f,' %loc_err)
                        fid.writelines('%.f,' %len(evoked.ch_names))
                        fid.writelines('%.3f,' %snr_dict[dfname][0])
                        fid.writelines('%.3f,' %reg)
                        fid.writelines('%s,' %prepcat)
                        fid.writelines('%s\n' %'DICS')
                        fid.close()
                        
                        plt.close('all')
#%%####################################### END (Good luck)####################################################

#plt.ion() 
#_, ax = plt.subplots(3, 1, figsize=(8, 8))  # create subplots
#plt.tight_layout()
#
#for ii in np.arange(0,101):
#    ax[0].cla()
#    ax[1].cla()
#    ax[2].cla()
#    epochs.copy().pick_types(meg='grad')[ii].average().plot(axes=ax[0], gfp=True, spatial_colors=True, titles="Epoch No: %d " % (ii + 1))  
#    epochs.copy().pick_types(meg='mag')[ii].average().plot(axes=ax[1], gfp=True, spatial_colors=True, titles="Epoch No: %d " % (ii + 1))  
#    epochs.copy().pick_types(meg='grad')[ii].average().plot_topomap(times=0.040, axes=ax[2], colorbar=False, outlines='head')
#    epochs.copy().pick_types(meg='mag')[ii].average().plot_topomap(times=0.040, axes=ax[3], colorbar=False, outlines='head')
#    plt.pause(0.25)
#    
#    
#    
##%% 
#plt.ion() 
#_, ax = plt.subplots(4, 1, figsize=(18, 8))  # create subplots
#plt.tight_layout()
#
#for ii in np.arange(30,45):
#    ax[0].cla()
#    ax[1].cla()
#    ax[2].cla()
#    ax[3].cla()
#    epochs.copy().drop([ii]).pick_types(meg='grad').average().plot(axes=ax[0], gfp=True, spatial_colors=True, titles="Epoch No: %d " % (ii + 1))  
#    epochs.copy().drop([ii]).pick_types(meg='mag').average().plot(axes=ax[1], gfp=True, spatial_colors=True, titles="Epoch No: %d " % (ii + 1))  
#    epochs.copy().drop([ii]).pick_types(meg='grad').average().plot_topomap(times=0.040, axes=ax[2], colorbar=False, outlines='head')
#    epochs.copy().drop([ii]).pick_types(meg='mag').average().plot_topomap(times=0.040, axes=ax[3], colorbar=False, outlines='head')
#    plt.pause(2.0)
        


