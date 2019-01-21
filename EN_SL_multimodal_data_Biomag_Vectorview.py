#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:24:16 2018

@author: Amit Jaiswal @ Elekta Oy (Helsinki)

=============================================>>
Master code for Source localization for Multimodal data (Vectorview system at BioMag)
datadir= /data/beamformer_data/Beamformer_share/multimodal/sub1/MEG/
Method: LCMV/DICS/Dfit/sLORETA/dSPM/MNE/RAPMUSIC
Stimuli: Auditory/Visual/SEF
Preprocessing methods: ICA/PCA optional
=============================================>>
"""
from __future__ import division
# Import dependencies
import matplotlib.pyplot as plt
#plt.rcParams['axes.facecolor']='white'
import numpy as np
from scipy.io import loadmat
import os
from os.path import exists
plt.close('all')
import sklearn as sk
from scipy.spatial import ConvexHull#, Delaunay
import mne
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from collections import OrderedDict
from mayavi import mlab
from itertools import product
#from sklearn.decomposition import PCA, FastICA 
from warnings import filterwarnings 
print(__doc__)

mne.set_log_level('WARNING')
filterwarnings('ignore')
plt.rcParams.update({'font.size':15})
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
        mne.viz.plot_alignment(raw.info, trans, subject=subject, subjects_dir=subjects_dir, fig=None,
                               surfaces=['head-dense', 'inner_skull'], coord_frame='head', show_axes=True,
                               meg=False, eeg='original', dig=True, ecog=True, bem=None, seeg=True,
                               src=mne.read_source_spaces(src), mri_fiducials=False,  verbose=True) 
    return src
#%% Run the loop
for prepcat in ['', '_sss','_tsss','_tsss-bp_2-95_ICAed']:#'', _tsss_mc '', '_sss', '',
    par={'site'      : 'Biomag', # (Vectorview system)
        'mode'       : 'MEG',  # Modality for analysis
        'datacat'    : 'multimodal',
        'chancat'    : 'all',  
        'prep'       : prepcat,                
        'badch'      : [],
        'apply_ica'  : 'no',
        'stimch'     : 'STI 014',
        'SLmeths'    : ['lcmv', 'dics', 'sloreta', 'dSPM', 'dfit', 'MNE', 'rap_music'],
        'dics'       : 'yes',
        'rap_music'  : 'yes',
        'calc_dip_VE': '',
        'multi_dipfit':'',
        'cort_const' : 'no',
        'do_dipfit'  : '',
        'other_SL'   : 'no', 
        'savefig'    : 'no',
        'savefig_res': '',
        'maxfilter'  : '',
        'notchfilter': 'yes',
        'bandpass'   : 'yes',
        'cal_src_s'  : 'yes',
        'cal_src_v'  : 'yes',
        'SL_cort'    : 'no',
        'gridres'    : 5.0,
        'calc_bem'   : 'yes',
        'bem_sol'    : 'inner_skull',#sphere/Xfit_bem/outer_skin
        'nc_meth'    : 'data',
        'visual'     : 'no',
        'powspect'   : 'yes',
        'save_resplot':'yes',
        'check_trial': 'yes',
        'browse'     : 'yes',
        'evoked_compare'  : '',
        'raw_psd_compare' : '',
        'more_plots' : '',
        'models_plot': '',
        'eog_ch'     : 'EOG 061',
        'ecg_ch'     : 'MEG 0141',
        'numtry'     : 1,
        'bin'        : 0.1,
        'cov_cut'    : [0.01, 98],
        'save_ave_beforeICA': 'yes',
        'save_ave_afterICA' : 'yes',
        'trialwin'   : [-0.500, 0.500],
        'ctrlwin'    : [-0.500, -0.0],
        'actiwin'    : [0.000, 0.500],
        'bpfreq'     : [2, 95],
        'icaext'     : ''}#-bp_2-95_ICAed
    
    #reg_form='snr_dict[dfname_stimcat]**4/500'
    reg_form = 'SNR**4/500'
    
    par['event_dict']=OrderedDict()
    par['event_dict']['VEF_UR']=1
    par['event_dict']['VEF_LR']=2
    par['event_dict']['AEF_Re']=3
    par['event_dict']['VEF_LL']=4 # Doubtful
    par['event_dict']['AEF_Le']=5
    par['event_dict']['VEF_UL']=8 # Doubtful
    par['event_dict']['SEF_Lh']=16
    par['event_dict']['SEF_Rh']=32
    st_len=len(par['event_dict'])

    print('Using backend %s for plots...........'%plt.get_backend())
#    act_dip=loadmat('/net/bonsai/home/amit/Documents/MATLAB/multimodal_biomag_diploc.mat')
#    act_dip=act_dip['multimodal_biomag_diploc'][0:8,[1,2,3]] # estimated from Xfit
    
    act_dip_ = loadmat('/net/bonsai/home/amit/Dropbox/multimodal_biomag_Xfit_results.mat')
    act_dip  = act_dip_['multimodal_biomag_Xfit_diploc'][:,3:6] 
    
    par['act_loc']=OrderedDict()    
    par['act_loc']['VEF_UR']=act_dip[0]
    par['act_loc']['VEF_LR']=act_dip[1]
    par['act_loc']['AEF_Re']=act_dip[2]
    par['act_loc']['VEF_LL']=act_dip[3]
    par['act_loc']['AEF_Le']=act_dip[4]
    par['act_loc']['VEF_UL']=act_dip[5]
    par['act_loc']['SEF_Lh']=act_dip[6]
    par['act_loc']['SEF_Rh']=act_dip[7]
#    mlab.points3d(act_dip[:,0],act_dip[:,1],act_dip[:,2])
#%% 
    data_path = '/data/beamformer_data/Beamformer_share/multimodal/sub1/MEG/'               
    fname     = data_path + 'multimodal_raw' + prepcat + par['icaext']+ '.fif'
    subjects_dir, subject = '/opt/freesurfer/subjects/', 'jukka_nenonen2'
    trans = subjects_dir + subject + '/mri/transforms/' + subject + '-trans.fif'
    # trans = '/opt/freesurfer/subjects/jukka_nenonen/mri/brain-neuromag/sets/jukka_nenonen-amit-131118-MNEicp-trans.fif'
    mrifile= subjects_dir + subject + '/mri/T1.mgz'
    surffile= subjects_dir + subject + '/bem/watershed/' + subject + '_brain_surface'
    out_dir='/net/bonsai/home/amit/Dropbox/BeamComp_Resultfiles/Biomag_multimodal_data/'
    out_path= out_dir + 'MNE/'
    resultfile = out_path + '/MNEp_Result-numtry' + str(par['numtry'])+ '-' + par['datacat'] + '-' + par['site'] + '_source_loc_file.csv'
    dfname=os.path.split(os.path.splitext(fname)[0])[1] 
    raw=mne.io.read_raw_fif(fname, allow_maxshield=False, preload=True, verbose=True)
    if not raw.info['projs']==[]:
         bads = ['MEG 0442']#, 'MEG 1121', 'MEG 0531', 'MEG 0623', 'MEG 0713', 'MEG 0733', 'MEG 1811']  # assign bad channel, if any
         raw.drop_channels(bads)
    events = mne.find_events(raw, stim_channel=par['stimch'], min_duration=0.001, shortest_event=1) # Read event
    events = events[events[:,1]==0]
    #plt.figure(), plt.plot(events[:,2])
    info=raw.info
    print(info)
    mne.viz.plot_events(events, sfreq=None, first_samp=0, color=None, event_id=par['event_dict'], 
                            axes=None, equal_spacing=True, show=True)
    
    #snr_dict=np.load(out_dir + par['site'] + '_' + par['datacat'] + '_SNR_' + par['chancat'] + '.npy').item()
    trigg_time=events[:,0]
    ISIs = [x-trigg_time[ii-1] for ii,x in enumerate(trigg_time)][1:]
    print('ISI is in between %d - %d' %(min(ISIs), max(ISIs)))
    
    #%% Set up pick list: EEG + MEG - bad channels (modify to your needs)
    if par['chancat']=='all':
        megchselect=True
    else:
        megchselect=par['chancat']
        
    picks = mne.pick_types(raw.info, meg=megchselect, eeg=False, stim=False, eog=False, ecg=False, emg=False, 
                           ref_meg='auto', misc=False, resp=False, chpi=False, exci=False, ias=False, 
                           syst=False, seeg=False, dipole=False, gof=False, bio=False, ecog=False, 
                           fnirs=False, include=[], exclude='bads', selection=None)
    #%% Browse data
    if par['browse']=='yesss':
        raw.plot(events=events, duration=10.0, start=0.0, n_channels=20, bgcolor='w', color=None, 
                 bad_color=(0.8, 0.8, 0.8), event_color='cyan', scalings=None, remove_dc=True, order=None,
                 show_options=False, title=dfname, show=True, block=False, highpass=None, lowpass=None,
                filtorder=4, clipping=None, show_first_samp=False, proj=True, group_by='type', 
                butterfly=False, decim='auto', noise_cov=None, event_id=None)
        raw.plot_psd(tmin=raw.times[0], tmax=raw.times[-1], fmin=0, fmax=300, proj=False, n_fft=None, picks=None, ax=None, 
                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
    
    #%% Apply filter if required
    if par['maxfilter']=='yes'and par['prep']=='':
        raw.fix_mag_coil_types()
        raw=mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.04), int_order=8, ext_order=3, 
                calibration=None, cross_talk=None, st_duration=None, st_correlation=0.98, coord_frame='head',
                destination=None, regularize='in', ignore_ref=False, bad_condition='error', head_pos=None,
                st_fixed=True, st_only=False, mag_scale=100.0, verbose=None)
        if par['more_plots']=='yes':
            raw.plot(events=events, title='Raw > MNE SSSed data plot')
            raw.plot_psd(tmin=0.0, tmax=None, fmin=0, fmax=300, proj=False, n_fft=None, picks=None, ax=None, 
                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
                        
    if par['notchfilter']=='yes' and par['icaext']=='':
        raw.notch_filter(50, filter_length='auto', phase='zero', picks=picks) # Notch filter
    if par['bandpass']=='yes' and par['icaext']=='':
        raw.filter(par['bpfreq'][0], par['bpfreq'][1], l_trans_bandwidth=min(max(2 * 0.01, 2), 2), 
                   h_trans_bandwidth=min(max(70 * 0.01, 2.), raw.info['sfreq'] / 2. - 70), 
                   filter_length='auto', phase='zero', picks=picks)
        if par['more_plots']=='yes':
            raw.plot(events=events, title='Raw> MNE SSSed data> Notch & badpassed data plot')
            raw.plot_psd(tmin=raw.times[0], tmax=raw.times[-1], fmin=par['bpfreq'][0], fmax=120, proj=False, n_fft=None, picks=None, ax=None, 
                    color='black', area_mode='std', area_alpha=0.33, n_overlap=0, dB=True, average=False,
                    show=True, n_jobs=1, line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
    
#commented on 08 Nov: raw.info.normalize_proj()# Re-normalize our empty-room projectors, so they are fine after subselection
    
#%% Save averaged evoked data before ICA
    if par['save_ave_beforeICA']=='yes---':
        for stimcat in par['event_dict'].keys():
            mne.write_evokeds(out_path + '/' + dfname + '-' + stimcat + '-bp_'+ str(int(par['bpfreq'][0]))+ '-'+ str(int(par['bpfreq'][1])) +'-ave.fif', 
                              mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],
                                         baseline=(None, 0), picks=picks, preload=True, flat=None, 
                                         proj=False, decim=1,reject_tmin=None, reject_tmax=None, 
                                         detrend= None, on_missing='error', reject_by_annotation=True,
                                         verbose=True)[stimcat].average())
            if par['more_plots']=='yes':
                mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average().plot(spatial_colors=True, gfp=True, xlim='tight',
                                         titles='Raw data>> Filtered between %s Hz' %par['bpfreq'] + '>> %s' %stimcat, verbose=True)
                
#%% ICA for DIGA data with tutorial code (use it)*******************************+++
    if 'ICAed' in prepcat and par['apply_ica']=='yes':
        n_components = 25   # if float, select n_components by explained variance of PCA
        method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
        decim = 3           # we need sufficient statistics, not all time points -> saves time
        random_state = 23   # fix a value to control the randomness in IC indexing
        reject = dict(mag=5e-12, grad=4000e-13) 
                                                
        ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
        print(ica)
        
        ica.fit(raw, picks=picks, start=None, stop=None, decim=decim, reject=reject, 
                reject_by_annotation=True, flat=None, verbose=True)
        print(ica)
        
        ica.plot_components(picks=range(0,n_components), ch_type=None, res=64, layout=None, vmin=None, vmax=None, 
                        cmap='RdBu_r', sensors=True, colorbar=False, title=None, show=True, 
                        outlines='head', contours=8, image_interp='bilinear', head_pos=None, inst=None)  # can you spot some potential bad guys?
        ica.plot_sources(raw)
        # look over props of some specific comps
        # ica.plot_properties(raw, picks=[0, 10, 19], psd_args={'fmax': 95.})
        title = 'Sources related to %s artifacts (red)'
        
#        ecg_inds, ecg_scores = ica.find_bads_ecg(mne.preprocessing.create_ecg_epochs(raw, ch_name=par['ecg_ch'], 
#                                                reject=reject, picks=picks, verbose=True), ch_name=par['ecg_ch'], 
#                                                threshold=0.020, start=None, stop=None, l_freq=1, h_freq=16, 
#                                                 method='correlation', reject_by_annotation=True, verbose=True)
        eog_inds, eog_scores = ica.find_bads_eog(mne.preprocessing.create_eog_epochs(raw, ch_name=par['eog_ch'], 
                                                         reject=reject, picks=None), ch_name=par['eog_ch'], threshold=1.0, 
                                                 start=None, stop=None, l_freq=None, h_freq=None, 
                                                 reject_by_annotation=True, verbose=None)
            
#        ecg_average = mne.preprocessing.create_ecg_epochs(raw, ch_name=par['ecg_ch'], 
#                                                         reject=reject, picks=picks, verbose=True).average()
#        ecg_average.plot(spatial_colors=True, gfp=True,  titles='Averaged %s artifacts' % 'ecg')
#        
        eog_average = mne.preprocessing.create_eog_epochs(raw, ch_name=par['eog_ch'], 
                                                         reject=reject, picks=picks).average()
        eog_average.plot(spatial_colors=True, gfp=True,  titles='Averaged %s artifacts' % 'eog')
        
        n_max_eog, n_max_ecg = 1, 0

#        ica.plot_scores(ecg_scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg', show=True)
        ica.plot_scores(eog_scores, exclude=eog_inds, title=title % 'eog', labels='eog', show=True)  
        
#        ica.plot_sources(ecg_average, exclude=ecg_inds)
        ica.plot_sources(eog_average, exclude=eog_inds)
        
#        ecg_inds = ecg_inds[:n_max_ecg]
        eog_inds = eog_inds[:n_max_eog]    
        
        manual_comp=[]       
    #    ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 95.})#,image_args={'sigma': 1.})
        
        print(ica.labels_)
        
#        ica.plot_overlay(ecg_average, exclude=ecg_inds, show=False)
        ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
        
#        ica.exclude.extend(ecg_inds)
        ica.exclude.extend(eog_inds)
        
        ica.exclude.extend(manual_comp)
        
        print('Excluded components = %s' %ica.exclude)
        
        del eog_average
        plt.close('all')
        
        ica.apply(raw)
        print('ICA applied (%s components removed)' %ica.exclude)
        
        raw.save(data_path + dfname + '-bp_'+ str(int(par['bpfreq'][0]))+ '-'+ str(int(par['bpfreq'][1])) + '_rmEOG_ICA.fif', overwrite=True)

#%% Save averaged evoked data after ICA
    if par['save_ave_afterICA']=='yes--':
        for stimcat in par['event_dict'].keys()[0:st_len]:
            mne.write_evokeds(out_path + '/' + dfname + '-' + stimcat +  '-bp_'+ str(int(par['bpfreq'][0]))+ '-'+ str(int(par['bpfreq'][1])) + '_rmEOGECG_ICA' + '-ave.fif', 
                              mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],
                                         baseline=(None, 0), picks=picks, preload=True, flat=None, 
                                         proj=False, decim=1,reject_tmin=None, reject_tmax=None, 
                                         detrend= None, on_missing='error', reject_by_annotation=True,
                                         verbose=True)[stimcat].average())
            if par['more_plots']=='yes':
                mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average().plot(spatial_colors=True, gfp=True, xlim='tight',
                                         titles='Raw data>> Filtered %s Hz >> ' %par['bpfreq'] + 
                                          '>> %s' %stimcat, verbose=True)

#%% Browse data and plot PSD before and after ICA
    if par['raw_psd_compare']=='yes':
        raw.plot_psd(tmin=0.0, tmax=raw.times[-1]/2, fmin=par['bpfreq'][0], fmax=par['bpfreq'][1], 
                                 proj=False, n_fft=None, picks=None, ax=None, color='black', area_mode='std', 
                                 area_alpha=0.33, n_overlap=0, dB=True, estimate='auto', average=False, show=True, n_jobs=1, 
                                 line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
        plt.title('Before_ICA')
        raw_copy.plot_psd(tmin=0.0, tmax=raw_copy.times[-1]/2, fmin=par['bpfreq'][0], fmax=par['bpfreq'][1], 
                                 proj=False, n_fft=None, picks=None, ax=None, color='black', area_mode='std', 
                                 area_alpha=0.33, n_overlap=0, dB=True, estimate='auto', average=False, show=True, n_jobs=1, 
                                 line_alpha=0.5, spatial_colors=True, xscale='linear', verbose=True)
        plt.title('After_ICA')
        raw.plot(events=events, duration=20.0, start=0.0, n_channels=30, bgcolor='w', color=None,
                     bad_color=(0.8, 0.8, 0.8), event_color='cyan', scalings=None, remove_dc=True, 
                     order='type', show_options=False, title=dfname + '_before_ICA', show=True, block=False, highpass=None, 
                     lowpass=None, filtorder=4, clipping=None, show_first_samp=False, proj=True, 
                     group_by='type', butterfly=False, decim='auto')
        raw_copy.plot(events=events, duration=20.0, start=0.0, n_channels=30, bgcolor='w', color=None,
                     bad_color=(0.8, 0.8, 0.8), event_color='cyan', scalings=None, remove_dc=True, 
                     order='type', show_options=False, title=dfname + '_after_ICA', show=True, block=False, highpass=None, 
                     lowpass=None, filtorder=4, clipping=None, show_first_samp=False, proj=True, 
                     group_by='type', butterfly=False, decim='auto')
#%% Compare evokeds before and after ICA (Joint plot)
    if par['evoked_compare']=='yes':
        for stimcat in par['event_dict'].keys()[0:st_len]:
            times=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,1.0,2.0,2.5,3.0,3.5,3.89]
            times=np.arange(-0.7,0.7,0.1)
            times='peaks'
            mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average().plot_joint(times=times, ts_args=dict(gfp=True),
                                         title='Raw data>> Filtered between %s Hz' %par['bpfreq'] + '>> %s' %stimcat)
            mne.Epochs(raw_copy, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average().plot_joint(times=times, ts_args=dict(gfp=True),
                                         title='Raw data>> Filtered %s Hz >> ' %par['bpfreq'] + 
                                         '%d EOG IC and ' %n_max_eog + '%d ECG IC removed' %n_max_ecg + '>> %s' %stimcat)
            
#%% Evoked grand amplitude compare plot
    if par['evoked_compare']=='yes':
        for stimcat in par['event_dict'].keys()[0:st_len]:
            mne.viz.plot_compare_evokeds(
                            [mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average(), 
                            mne.Epochs(raw_copy, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                            baseline=(None, 0), picks=picks, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, 
                            verbose=True)[stimcat].average()], 
                                         picks=[], gfp=True, colors=None, linestyles=['-'], 
                                         styles=None, vlines=[0.0], ci=0.95, truncate_yaxis=False, ylim={}, 
                                         invert_y=False, axes=None, title='Evoked Grand compare before(1), after(2) ICA for %s' %stimcat, show=True)
                            
#%% PSD topomap 
    if par['evoked_compare']=='yes':
        for stimcat in par['event_dict'].keys()[0:st_len]:
            bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 45, 'Gamma'), (30, 45, 'H_Gamma')]
            mne.viz.plot_epochs_psd_topomap(mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                        baseline=(None, 0), picks=picks, preload=True,  
                        flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                        on_missing='error', reject_by_annotation=True, 
                        verbose=True)[stimcat],
                            bands=bands, vmin=None, vmax=None, tmin=None, tmax=None, 
                            proj=False, bandwidth=None, adaptive=False, low_bias=True, 
                            normalization='length', ch_type=None, layout=None, cmap='RdBu_r',
                            agg_fun=None, dB=True, n_jobs=1, normalize=False, cbar_fmt='%0.3f', 
                            outlines='head', axes=None, show=True, verbose=True)
            plt.title('Before ICA- %s' %stimcat)
            mne.viz.plot_epochs_psd_topomap(mne.Epochs(raw_copy, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                        baseline=(None, 0), picks=picks, preload=True,  
                        flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                        on_missing='error', reject_by_annotation=True, 
                        verbose=True)[stimcat],
                            bands=bands, vmin=None, vmax=None, tmin=None, tmax=None, 
                            proj=False, bandwidth=None, adaptive=False, low_bias=True, 
                            normalization='length', ch_type=None, layout=None, cmap='RdBu_r',
                            agg_fun=None, dB=True, n_jobs=1, normalize=False, cbar_fmt='%0.3f', 
                            outlines='head', axes=None, show=True, verbose=True)
            plt.title('After ICA- %s' %stimcat)
    
    if 'raw' in locals() and  par['apply_ica']=='yes':
        del raw
#%% Epoch the ICA applied data
#    eog_events = mne.preprocessing.find_eog_events(raw)
#    n_blinks = len(eog_events)
#    # Center to cover the whole blink with full duration of 0.5s:
#    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
#    duration = np.repeat(0.5, n_blinks)
#    annot = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
#                            orig_time=raw.info['meas_date'])
#    raw.set_annotations(annot)# in version 0.17
#    print(raw.annotations)  # to get information about what annotations we have
#    raw.plot(events=eog_events)  # To see the annotated segments.
    
    reject = dict(grad=7000e-13, # T / m (gradiometers)
              mag=7e-12, # T (magnetometers)
              eog=250e-6) # V (EOG channels)
    epochs = mne.Epochs(raw, events, par['event_dict'], par['trialwin'][0], par['trialwin'][1],  
                        baseline=(par['trialwin'][0], 0), picks=None, preload=True, reject=None, #reject,#None, 
                        flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                        on_missing='error', reject_by_annotation=True, verbose=True)
    ### epochs.plot(events=events)
    for stimcat in list(par['event_dict'].keys())[0:8]:
        epochs[stimcat].average().plot(spatial_colors=True, titles=stimcat,gfp=True, time_unit='ms')
#        plt.pause(0.25)
#        plt.tight_layout
#        plt.pause(0.25)
#        plt.savefig('/net/qnap/data/rd/ChildBrain/JYU_Jan_PPT_materials' + '/' + dfname + '-' + stimcat + '_EvokedPlot_numtry'+ str(par['numtry'])+ '.png', 
#                                dpi=None, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, 
#                                format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
#        plt.close()
            
#%% Compute Source Space ...............>>
    src = define_source_space_and_plot(mode='vol', plot_alignment='no', gridres=5.0, 
                                       spacing='ico4', surf_type='pial', mindist=2.5, exclude=10.0)
    if 'raw' in locals():
        del raw
#%% BEM modeling......................>>>>
    if par['calc_bem']=='yes':
        if par['bem_sol']=='sphere':
            bem=mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=None, info=epochs.info, verbose=True) 
        elif par['bem_sol']=='inner_skull':#'brain': # outer_skin
            model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.33,), 
                                       subjects_dir=subjects_dir, verbose=True)
#            mne.write_bem_surfaces(subjects_dir + subject + '/bem/' + subject + '-' + par['bem_sol'] + 
#                                   '-bem.fif', model)
            mne.write_bem_surfaces('/neuro/databases/bem/' + subject + '-' + par['bem_sol'] + '-bem.fif', model)
            bem = mne.make_bem_solution(model) # bem-solution
#            mne.write_bem_solution(subjects_dir + subject + '/bem/' + subject + '-' + par['bem_sol'] + 
#                                   '-bem-sol.fif', bem)
            mne.write_bem_solution('%s%s-%s-bem-sol.fif' %('/neuro/databases/bem/', subject, par['bem_sol']), bem)
        elif par['bem_sol']=='Xfit_bem':
            bem=mne.read_bem_solution('/neuro/databases/bem/phantom_sphere-bem-sol.fif', verbose=True)
            mne.write_bem_solution(subjects_dir + subject + '/bem/' + subject + '-' + par['bem_sol'] + 
                                   '-bem-sol.fif', bem)  
    if par['more_plots']=='yessss' or par['models_plot']=='yess':
        mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='pial', 
                                 src=src, slices=range(73,193,5), orientation='sagittal')
        
    surf_h=mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) # transform            
#%% Forward solution..................>>  
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                    meg=True, eeg=False, mindist=2.5, n_jobs=1)#n_jobs=2
    print("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)
#    mne.write_forward_solution(subjects_dir + subject + '/' + dfname + '_volume-' + 
#                               str(par['gridres'])+ 'mm_fwd.fif', fwd, overwrite=True) 
#%% Apply LCMV Beamformer 
    cov_meths = ['empirical', 'shrinkage']#, 'ledoit_wolf', 'oas', 'shrunk', 'diagonal_fixed']
    weight_norms = ['nai']#, 'unit-noise-gain']
    for weight_norm, cov_meth, stimcat in product(weight_norms, cov_meths, list(par['event_dict'].keys())):
    # for stimcat in par['event_dict'].keys()
        if 'VEF' in stimcat: # ............for evoked
            par['ctrlwin']=[-0.200, -0.050]
            par['actiwin']=[0.050, 0.200]
            #reg=1.0
        elif 'AEF' in stimcat:
            par['ctrlwin']=[-0.130, -0.020]
            par['actiwin']=[0.020, 0.130]
            #reg=1.0
        elif 'SEF' in stimcat:
            par['ctrlwin']=[-0.100, -0.010]
            par['actiwin']=[0.010, 0.100]
            #reg=1.0
        dfname_stimcat  = dfname +  '_' + stimcat
#        cov_meth = 'shrinkage'#'empirical' #
#        weight_norm = 'nai'#'unit-noise-gain'
        badtrls = []
        AEF_Re_bads = ['MEG 1323', 'MEG 1322', 'MEG 1333', 'MEG 1332', 'MEG 1343', 'MEG 1341', 'MEG 1443', 'MEG 2423', 'MEG 2612', 'MEG 2613'] # from Xfit
        
        epochs_stimcat =  epochs[stimcat]
        # Find trial variance && z-score outliers and index them
        trl_var = np.empty((0,1), 'float')
        #trl_zscore = np.empty((0,1), 'float')
        trlindx = np.arange(0,len(epochs_stimcat))
        for trnum in range(len(epochs_stimcat)):
            trl_var= np.vstack((trl_var, max(np.var(np.squeeze(epochs_stimcat[trnum].get_data()), axis=1))))
        lim1 = (trl_var < np.percentile(trl_var, par['cov_cut'][0], interpolation='midpoint')).flatten()
        lim2 = (trl_var > np.percentile(trl_var, par['cov_cut'][1], interpolation='midpoint')).flatten()
        outlr_idx = trlindx[lim1].tolist() + trlindx[lim2].tolist()
        plt.figure(), plt.scatter(trlindx, trl_var, marker='D'), plt.ylabel('Max. Variance accros channels-->')
        plt.scatter(outlr_idx, trl_var[outlr_idx],s=300, marker='.', facecolors='none', edgecolors='r', 
                    linewidth=2), plt.xlabel('Trial number-->')
        plt.scatter(badtrls, trl_var[badtrls],marker='.', facecolor=None)
        plt.ylim(min(trl_var), max(trl_var)), plt.title('        Max. variance distribution')          
        bad_trials = np.union1d(badtrls, outlr_idx)
        print('\n%d trials removed because of high variance: %s\n' %(len(bad_trials), str(bad_trials)))
        print('\nRemaining #trials = %d - %d = %d trials .........\n' %(len(epochs_stimcat), len(bad_trials),len(epochs_stimcat)-len(bad_trials)))
        epochs_stimcat.drop(bad_trials, reason='high variance', verbose=True) # added on 29 nov
        epochs_stimcat.pick_types(meg=True)
              
        print('\ndfname = %s\n\nBaseline window= %s\n\nActive window = %s\n'
              %(dfname_stimcat, str(par['ctrlwin']), str(par['actiwin'])))
        print('Covariance method = %s\n\nweight_norm= %s\n'%(cov_meth, weight_norm))
        noise_cov = mne.compute_covariance(epochs_stimcat, tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1], method=cov_meth, verbose=True) # shrunk
        data_cov = mne.compute_covariance(epochs_stimcat, tmin=par['actiwin'][0], tmax=par['actiwin'][1], method=cov_meth, verbose=True)   # shrunk
        #noise_cov['cov_meth'] = noise_cov['method'] # just add comment
#        data_cov = mne.compute_covariance(epochs_stimcat, tmin=0.0, tmax=par['trialwin'][1], method='empirical')   # shrunk
        noise_cov.save('%s_%s_noise_cov.fif'%(fname[:-4],cov_meth))
        data_cov.save('%s_%s_data_cov.fif'%(fname[:-4],cov_meth))
        #noise_cov=mne.cov.regularize(noise_cov, info, mag=0.9, grad=0.9, eeg=0.1, exclude='bads', proj=True, verbose=True)
        #noise_cov.plot(epochs_stimcat.info, show_svd=True, proj=True)
        #data_cov.plot(epochs_stimcat.info, show_svd=True, proj=True)
        if par['more_plots']=='yess':
#        with plt.xkcd():
            fig=plt.figure()
            fig.suptitle("Noise and data covariance plot for %s" %stimcat, fontsize=15)
            ax1 = plt.subplot(1,3,1)
            plt.imshow(noise_cov.data) #, cmap='jet', aspect='auto'  # , vmin=-100, vmax=100, cmap='hot'
            ax1.set_title("Noise cov.(Nc)",  fontsize=15) 
            ax1.set_xlabel('channels', fontsize=15)
            ax1.set_ylabel('channels', fontsize=15)
#                    plt.axis('off')
            plt.colorbar(orientation='horizontal')
            ax2 = plt.subplot(1,3,2, sharex=ax1)              
            plt.imshow(data_cov.data)
            ax2.set_title("Data cov. (Dc)",  fontsize=15)
            ax2.set_xlabel('channels', fontsize=15)
            ax2.set_ylabel('channels', fontsize=15)
#                    plt.axis('off')
            plt.colorbar(orientation='horizontal')
            ax3 = plt.subplot(1,3,3, sharex=ax1)
            plt.imshow(((data_cov.data-noise_cov.data)/noise_cov.data))
            ax3.set_title("(Dc-Nc)/Nc",  fontsize=15)
            ax3.set_xlabel('channels', fontsize=15)
            ax3.set_ylabel('channels', fontsize=15)
#                    plt.axis('off')
            plt.colorbar(orientation='horizontal')
            plt.pause(0.1)
            manager=plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.pause(0.1)
            plt.tight_layout()
            plt.pause(0.1)
            plt.subplots_adjust(top=0.69, bottom=0.01)
            plt.savefig(out_path + '/' + dfname + '-' + stimcat + '_CovPlot_induced_numtry'+ str(par['numtry'])+ '.png', 
                                dpi=None, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, 
                                format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
        
        evoked = epochs_stimcat.average()
        # evoked.plot(spatial_colors=True, gfp=True, time_unit='ms')
        evoked = evoked.crop(par['actiwin'][0], par['actiwin'][1])
        
        cov_rank = None if epochs_stimcat.info['proc_history']==[] else int(epochs_stimcat.info['proc_history'][0]['max_info']['sss_info']['nfree'])
        inverse_operator=mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov, 
                                                                rank=cov_rank, loose=1, depth=0.199, verbose=True)
        snr, snr_est = mne.minimum_norm.estimate_snr(evoked, inverse_operator, verbose=True)
        if par['more_plots']=='yes':
            figname= dfname + '-' + stimcat
            plt.figure(figname + '_snr&snr_est') 
            plt.plot(evoked.times, snr, 'r', label='snr'),  plt.hold(True), plt.plot(evoked.times, snr_est, 'g', label='snr_est')
            plt.xlim([evoked.times[-0], evoked.times[-1]])
            plt.legend(loc='upper right', fontsize=None, shadow=False, framealpha=0.3)
            plt.suptitle(figname + '_snr&snr_est') 
    #        plt.savefig(out_path + figname + '_snr&snr_est', facecolor='w', edgecolor='w', orientation='landscape', bbox_inches='tight', pad_inches=0.2)
        peak_ch, peak_time = evoked.get_peak(ch_type='mag')
        tstep=1000/(evoked.info['sfreq']*1000)
        tp = int(peak_time//tstep - evoked.times[0]//tstep)        
        SNR=snr[tp]
        #mne.viz.plot_snr_estimate(evoked, inverse_operator, show=True)
        #plt.ylim(0,30)
        #reg=eval(reg_form)
        for reg in (0.05, eval(reg_form)):
            rank = cov_rank # np.sum(np.linalg.svd(noise_cov.data)[1]>1e-25)-2 #np.linalg.matrix_rank(evoked.copy().crop(par['ctrlwin'][0], par['ctrlwin'][1]).data), 
            # plt.figure('svd plot'), plt.plot(np.linalg.svd(noise_cov.data)[1])
#            filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=reg, noise_cov=noise_cov,
#                                                      pick_ori='max-power', rank=rank,
#                                                      weight_norm=weight_norm, reduce_rank=True, verbose=True)
            #weight_norm='nai'#'unit-noise-gain'
            if 'filters' in locals():
                del filters
            reducerank = 0    
            while not 'filters' in locals():
                try:
                    filters = mne.beamformer.make_lcmv(epochs_stimcat.info, fwd, data_cov, reg=reg, 
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
                                    
            stc = mne.beamformer.apply_lcmv(evoked, filters, max_ori_out='signed', verbose=True)
#           stc.data[:, :] = np.abs(stc.data)
            #stc.crop(0.01, stc.times[-1])
            stc=np.abs(stc)
            #stc.crop(t1, stc.times[-1])
            #stc.save(out_path + '/'+ dfname + '-' + stimcat + '_lcmv_volumetric_' + name)
            src_peak, t_peak = stc.get_peak()
            timepoint = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)
            est_loc=fwd['src'][0]['rr'][src_peak]*1000 # in mili meter
            loc_err=np.sqrt(np.sum(np.square(par['act_loc'][stimcat]-est_loc))); # Calculate loc_err

            print('Act_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %par['act_loc'][stimcat]) 
            print('Est_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %np.around(est_loc,1))
            print('Peak_value for %s-' %dfname + '%s' %stimcat + '= %.2f\n' % stc.data.max())
            print('Loc_error for %s-' %dfname + '%s' %stimcat + '= %.1fmm\n' %loc_err)
            
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
            if len(idx[idx==1])==len(clust_data1):
                xxx=1
            else:
                xxx=0            
            locs1 = fwd['source_rr'][idx==xxx,:]
            n_act_vert = len(locs1)
            totalVolume2 = n_act_vert*(par['gridres']**3)
            locs2 = fwd['source_rr'][idx==0,:]
            cntrd_locs1 = np.array([[locs1[:,0].mean(), locs1[:,1].mean(), locs1[:,2].mean()]]);
            cntrd_locs1mm = cntrd_locs1*1000
            loc_err2 =  np.sqrt(np.sum(np.square(par['act_loc'][stimcat]-(cntrd_locs1*1000)))) 
            kk=np.array([0])
            for ii in range(n_act_grid):
                kk = np.vstack((kk,np.sqrt(np.sum(np.square(locs1[ii,:]-cntrd_locs1)))*1000))
            meandist =  sum(kk)[0]/(n_act_grid-1)  
                     
            while len(locs1)<=4:
                locs1 = np.vstack((locs1, locs1[0,:]))
            hull = ConvexHull(locs1, incremental=False, qhull_options='QJ')
            tris1 = hull.simplices
            totalVolume = hull.volume *1000*1000*1000
            totalArea   = hull.area *1000*1000
            # [totalVolume3, totalArea3] = stlVolume(locs1,tris1)
            print('Total volume = %.2fmm3 \nTotal area = %.2fmm2' %(totalVolume, totalArea))                
            print('No. of active voxels = %d \nActual total volume = %.2fmm3' %(n_act_grid, totalVolume2))
            
            if par['more_plots']=='yes':  # 3D plot using mayavi and animate              
                stc_loc_mm, est_loc_mm = par['act_loc'][stimcat]/1000, est_loc/1000
                mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700))
                #mlab.points3d(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], mode='sphere', scale_factor=.001, color=(0.8,0.9,0.8))
                mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0.8,0.9,0.8))
                mlab.points3d(0.000,0.000,0.000, mode='sphere', scale_factor=.005, color=(0,0,1))
                mlab.points3d(stc_loc_mm[0], stc_loc_mm[1], stc_loc_mm[2], mode='sphere', scale_factor=0.005, color=(0,1,0))
                mlab.points3d(est_loc_mm[0], est_loc_mm[1], est_loc_mm[2], mode='sphere', scale_factor=0.005, color=(1,0,0))
                mlab.points3d(cntrd_locs1[:,0], cntrd_locs1[:,1], cntrd_locs1[:,2], mode='sphere', scale_factor=0.005, color=(0.9,0.8,0.9))
                mlab.points3d(locs1[:,0], locs1[:,1], locs1[:,2],mode='sphere', scale_factor=0.0005, color=(1,1,0))
                mlab.triangular_mesh(locs1[:,0], locs1[:,1], locs1[:,2], tris1, color=(0.9,0.1,0.1), transparent=True, opacity=0.4)
                #mlab.text3d(0.000,0.000,0.000, 'Origin', scale=0.005, color=(0,0,1))
                #mlab.text3d(stc_loc_mm[0], stc_loc_mm[1], stc_loc_mm[2], 'TrueLoc.', scale=0.005, color=(0,1,0))
                #mlab.text3d(est_loc_mm[0], est_loc_mm[1], est_loc_mm[2], 'EstLoc.', scale=0.005, color=(1,0,0))
                #mlab.text3d(cntrd_locs1[:,0][0], cntrd_locs1[:,1][0], cntrd_locs1[:,2][0], 'Centroid', scale=0.005, color=(0.9,0.8,0.9))
                helmet = mne.surface.get_meg_helmet_surf(info=evoked.info, trans=None, verbose=True)
                mlab.triangular_mesh(helmet['rr'][:,0], helmet['rr'][:,1], helmet['rr'][:,2], helmet['tris'], representation='wireframe', 
                                     mode='sphere', opacity=0.9, scale_factor=.1, color=(0,0.6,1))
                mlab.roll(-90)
                #a = anim() # Starts the animation.
                mlab.show()
            
#%          #Open a file and type location amp and error
            #os.mkdir('/neuro/data/BeamComp/Phantom/Aston/' +'Phantom_MNE_LCMV_loop_result-numtry'+ str(par['numtry']), 0777)
            fid = open(resultfile, 'a+')
            if stimcat=='VEF_UR' and reg==0.05:
                fid.writelines('\n%s\n' %reg_form)
            fid.writelines('%s,'   %dfname)
            fid.writelines('%s,'   %stimcat)
            fid.writelines('%.2f,' %est_loc[0])
            fid.writelines('%.2f,' %est_loc[1])
            fid.writelines('%.2f,' %est_loc[2])
            fid.writelines('%.2f,' %stc.data.max())
            fid.writelines('%.2f,' %loc_err)
            fid.writelines('%.2f,' %np.sqrt(np.sum(np.square([0,0,0]-par['act_loc'][stimcat]))))
            fid.writelines('%.2f,' %np.sqrt(np.sum(np.square([0,0,0]-est_loc))))
            fid.writelines('%d,'   %evoked.nave)
            fid.writelines('%d,'   %len(evoked.ch_names))
            fid.writelines('%.3f,' %SNR)
            fid.writelines('%.3f,' %reg)
            fid.writelines('%s,'   %str(rank))
            fid.writelines('%s,'   %'LCMV')
            fid.writelines('%.2f,' %totalVolume)
            fid.writelines('%.2f,' %totalArea)
            fid.writelines('%d,'   %n_act_grid)
            fid.writelines('%.2f,' %PSVol) 
            fid.writelines('%.2f,' %loc_err2)
            fid.writelines('%.2f,' %cntrd_locs1[0][0])
            fid.writelines('%.2f,' %cntrd_locs1[0][1])
            fid.writelines('%.2f,' %cntrd_locs1[0][2])
            fid.writelines('%.2f,' %meandist)
            fid.writelines('%s,'   %cov_meth)
            fid.writelines('%d,'   %(t_peak*1000))
            fid.writelines('%s,\n' %weight_norm)
            fid.close()
            
            if par['more_plots']=='yes':
                plt.figure()
                ts_show = -50
                plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
                plt.title('LCMV STC plot for %s-'%dfname + '%s'%stimcat + ' for %d largest sources'%abs(ts_show))
                plt.xlabel('time (ms)')
                plt.ylabel('%s value' % 'LCMV' + '@ Reg_par= %.5f' %reg)
                plt.show()
                plt.savefig(out_path + figname + '_STCplot.png', facecolor='w', edgecolor='w', 
                            orientation='landscape', bbox_inches='tight', pad_inches=0.2)
            
            if par['more_plots']=='yes' and not par['SL_cort']=='yes': # Volumetric plot
                thresh = stc.data.max()*0.50
                img=mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc, fwd['src'], dest='mri', mri_resolution=False)
                plot_stat_map(index_img(img, timepoint), mrifile, threshold=thresh)
                plt.suptitle('%s-'%dfname + '%s'%stimcat + 'LCMV (tpeak=%.3f s.)' % stc.times[timepoint] + 
                             'PeakValue= %.3f\n' % stc.data.max() + ' / Reg= %.3f' % reg + 'Est_loc= %.1f' % est_loc[0] + 
                             ', %.1f' % est_loc[1]+', %.1f ' % est_loc[2] + '/ Loc_err= %.2f mm' % loc_err, 
                             fontsize=15, color='white')
#                plt.pause(1.0)
#                manager=plt.get_current_fig_manager()
#                manager.window.showMaximized()
                if par['save_resplot']=='yes':
                    plt.savefig(out_path + figname + '_Sourceplot.png', 
                                        dpi=100, facecolor='w', edgecolor='w', orientation='landscape', papertype=None, 
                                        format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
                os.remove('lcmv_inverse.nii.gz')
            elif par['more_plots']=='yes' and not par['SL_cort']=='yes': # Cortically constrainned plot
                brain=stc.plot(subject=subject, surface='inflated', hemi='both', colormap='auto', time_label='auto', 
                     smoothing_steps=10, transparent=None, alpha=1.0, time_viewer=False, subjects_dir=subjects_dir, 
                     figure=None, views='lat', colorbar=True, clim='auto', cortex='classic', size=800, 
                     background='black', foreground='white', initial_time=t_peak, time_unit='s', 
                     backend='auto', spacing='oct5')
                ##brain.add_foci(tuple(est_loc), coords_as_verts=True, scale_factor=0.5, hemi='lh', color='r')
            
# Cortically constrained plot 
#            brain = stc.plot(hemi='both', subjects_dir=subjects_dir, initial_time=0.1, time_unit='s')
#            brain.show_view('lateral')
#            for color, vertex in zip(colors, max_voxs):
#                brain.add_foci([vertex], coords_as_verts=True, scale_factor=0.5,
#                               hemi='lh', color=color)
            plt.close('all')
            
#%% Apply DICS:
    if par['dics']=='':
        ICappliedondata='yes'
        if ICappliedondata=='yes':
            icaed='ICAed_data_'
        else:
            icaed=''
        
        for stimcat in par['event_dict'].keys()[0:st_len]:
            if 'VEF' in stimcat:
                par['ctrlwin']=[-0.200, -0.050]
                par['actiwin']=[0.050, 0.200]
            elif 'AEF' in stimcat:
                par['ctrlwin']=[-0.150, -0.020]
                par['actiwin']=[0.020, 0.150]
            elif 'SEF' in stimcat:
                par['ctrlwin']=[-0.100, -0.010]
                par['actiwin']=[0.010, 0.100]
            fbandwidth=[par['bpfreq'][0], 48]
            snr=2
            reg = 1. / snr ** 2
            REG=0.05
            
            if par['more_plots']=='yesss':
                epochs[stimcat].plot_psd(fmin=par['bpfreq'][0], fmax=par['bpfreq'][1])
            # Computing the data and noise cross-spectral density matrices
            # The time-frequency window was chosen on the basis of spectrograms from
            # example time_frequency/plot_time_frequency.py
            noise_csd = mne.time_frequency.csd_epochs(epochs[stimcat], mode='multitaper', tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1],
                                   fmin=fbandwidth[0], fmax=fbandwidth[1])
            data_csd = mne.time_frequency.csd_epochs(epochs[stimcat], mode='multitaper', tmin=par['actiwin'][0], tmax=par['actiwin'][1],
                                  fmin=fbandwidth[0], fmax=fbandwidth[1])
            # evoked = epochs.average()
            evoked=epochs[stimcat].average()
            evoked=evoked.crop(par['actiwin'][0], par['actiwin'][1])
            # Compute DICS spatial filter and estimate source time courses on evoked data
            stc = dics(evoked, fwd, noise_csd, data_csd, reg=reg)
            
            plt.figure()
            ts_show = -50  # show the 200 largest responses
            plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
            plt.title('DICS STC plot for %s-'%dfname + '%s'%stimcat + ' for %d largest sources'%abs(ts_show))
            plt.xlabel('time (ms)')
            plt.ylabel('%s value' % 'DICS' + '@Reg= %.1f' %reg)
            plt.show()
            
#            tcrop=0.0
#            stc.crop(0.020, stc.times[-1])
            stc.save(out_path + '/'+ dfname + '_' + icaed + stimcat + '_dics_volumetric')
        
            img=mne.save_stc_as_volume(out_path + '/' + dfname + '_' + icaed + stimcat + '_dics_inverse.nii.gz', stc, 
                                       fwd['src'], dest='mri', mri_resolution=False)
            
            t_peak=stc.get_peak()[1]
            
            stc_peak=stc.get_peak() # stc data peak timepoint and vertex
            
            thresh = stc.data.max()*70/100
            timepoint = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)
            
            est_loc=fwd['src'][0]['rr'][stc_peak[0]]*1000 # in mili meter
            loc_err=np.sqrt(np.sum(np.square(par['act_loc'][stimcat]-est_loc))); # Calculate loc_err
            
            print('Act_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %par['act_loc'][stimcat]) 
            print('Est_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %np.around(est_loc,1))
            print('Loc_error for %s-' %dfname + '%s' %stimcat + '= %.1fmm\n' %loc_err)
            print('Peak_value for %s-' %dfname + '%s' %stimcat + '= %.2f\n' % stc.data.max())
            
            resultfile = resultfile
            fid = open(resultfile, 'a')
            fid.writelines(dfname + '-chan_' + par['chancat']+ '-' + icaed + '-reg_'+ str(reg) + '-' + stimcat + '-DICS'
                           '     :  %s' %np.round(est_loc,2) + '  %.3f' %stc.data.max() + '  %.2f' % loc_err + '     %d\n' %(t_peak*1000))
         
            fid.close()
            
            # Plot brain in 3D with PySurfer if available (for corticaly constrained)
            if par['SL_cc']=='yesss':
                brain = stc.plot(hemi='rh', subjects_dir=subjects_dir, initial_time=0.1, time_unit='s')
                brain.show_view('lateral')
                brain.save_image('DICS_map.png')
                
            plot_stat_map(index_img(img, timepoint), mrifile, threshold=thresh, 
                  title='%s-'%dfname + '%s'%stimcat + ' / DICS (tpeak=%.3f s.)\n' % stc.times[timepoint] + 
                  'PeakValue= %.3f' % stc.data.max() + ' / Reg= %.3f\n' % reg + 
                  'Est_loc= %.1f' % est_loc[0]+', %.1f' % est_loc[1]+', %.1f ' % est_loc[2] +
                  '/ Loc_err= %.2f mm' % loc_err)  
            #                plt.pause(1.0)
#            manager=plt.get_current_fig_manager()
#            manager.window.showMaximized()
            
            
            
#%% Rap Music:                 
    if par['rap_music']=='yes--':
        
        ICappliedondata='yes'
        if ICappliedondata=='yes':
            icaed='ICAed_data_'
        else:
            icaed=''
        
        for stimcat in par['event_dict'].keys()[0:st_len]:
            if 'VEF' in stimcat:
                par['ctrlwin']=[-0.200, -0.050]
                par['actiwin']=[0.050, 0.200]
            elif 'AEF' in stimcat:
                par['ctrlwin']=[-0.150, -0.020]
                par['actiwin']=[0.020, 0.150]
            elif 'SEF' in stimcat:
                par['ctrlwin']=[-0.100, -0.010]
                par['actiwin']=[0.010, 0.100]
            n_dipoles=1
            idx='gof' # 'amplitude' ''
#            snr=2
#            reg = 1. / snr ** 2
        
            evoked=epochs[stimcat].average().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
            #evoked.pick_types(meg=True, eeg=False)
            
            noise_cov = mne.compute_covariance(epochs[stimcat], tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1], method='empirical') # shrunk
            
            dipoles, residual = rap_music(evoked, fwd, noise_cov, n_dipoles=n_dipoles, return_residual=True, verbose=True)
            
            if par['more_plots']=='yess':
                transs = fwd['mri_head_t']
                mne.viz.plot_dipole_locations(dipoles, transs, subject=subject, subjects_dir=subjects_dir, mode='orthoview', 
                                              coord_frame='head', idx=idx, show_all=True, ax=None, block=False, show=True, verbose=True)
                fontdict={'fontsize': 15, 'fontweight' : 1,
                          'verticalalignment': 'bottom', #(u'top', u'bottom', u'center', u'baseline')
                          'horizontalalignment': 'right'}
                plt.title('%s ' %stimcat, fontdict=fontdict)
                
                # Plot dipole amplitude
                mne.viz.plot_dipole_amplitudes(dipoles, colors=None, show=True)
                plt.title('Dipole amplitude plot for %s- ' %dfname + '%s' %stimcat)
                # Plot the evoked data and the residual.
                ylim=dict(grad=[-300, 300], mag=[-1000, 1000], eeg=[-6, 8])
                evoked.plot(ylim=None, spatial_colors=True, titles='Evoked plot for %s- ' %dfname + '%s' %stimcat)
                residual.plot(ylim=None, spatial_colors=True, titles='Residual plot for %s- ' %dfname + '%s' %stimcat)
            
            est_loc=dipoles[0].pos[0]*1000
            #loc_err=np.sqrt(np.sum(np.square(act_dip[par['nD']-1]-est_loc))); # Calculate loc_err
            
#            print('Act_diploc for %s' %dfname + '= %s' %act_dip[par['nD']-1,:]) 
            print('Est_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %np.around(est_loc,1))
#            print('Loc_error for %s' %dfname + '= %.1fmm\n' %loc_err)
            print('Dip_amplitude for %s-' %dfname + '%s' %stimcat + '= %.2fnAm' %(dipoles[0].amplitude.max()*10e8))

            resultfile = resultfile
            fid = open(resultfile, 'a')
            fid.writelines(dfname + '-chan_' + par['chancat']+ '-' + icaed + 'ncov_' + par['nc_meth'] + '-idx_'+ idx + '-' + stimcat + '-rapmusic'
                           '     :  %s' %np.round(est_loc,2) + '  %.2f' %dipoles[0].amplitude.max() + '  %.2f\n' % 0.00)
            fid.close()
            
#%% Dipole fitting (Single dipole)  # Using the single highest peak from data 
    if par['do_dipfit']=='yes':
        ICappliedondata='yes'
        if ICappliedondata=='yes':
            icaed='ICAed_data_'
        else:
            icaed=''
        
        for stimcat in par['event_dict'].keys()[0:st_len]:
            if 'VEF' in stimcat:
                par['ctrlwin']=[-0.200, -0.050]
                par['actiwin']=[0.050, 0.200]
            elif 'AEF' in stimcat:
                par['ctrlwin']=[-0.150, -0.020]
                par['actiwin']=[0.020, 0.150]
            elif 'SEF' in stimcat:
                par['ctrlwin']=[-0.100, -0.010]
                par['actiwin']=[0.010, 0.100]
            idx='gof' # 'amplitude' ''
            peakspan=5 # samples both sides of t_peak
        
            evoked=epochs[stimcat].average().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
            data_cov = mne.compute_covariance(epochs[stimcat], tmin=par['actiwin'][0], tmax=par['actiwin'][1], method='empirical')   # shrunk

            t_peak=evoked.get_peak(ch_type='grad')[1] # Taking the peak point from evoked data
            # t_peak= any time of interest 
            evoked_full=evoked.copy()
            evoked = evoked.crop(t_peak - peakspan/evoked.info['sfreq'], t_peak + peakspan/evoked.info['sfreq'])

            dipoles = mne.fit_dipole(evoked, data_cov, bem, trans=trans, min_dist=0.0, 
                                     n_jobs=1, pos=None, ori=None, verbose=True)[0]
            
            # find the time point with highest GOF to plot
            best_idx = np.argmax(dipoles.gof)
            best_time=dipoles.times[best_idx]
            
            print('Highest GOF %0.1f%% at t=%0.1f ms with confidence volume %0.1f cm^3'
                      % (dipoles.gof[best_idx], best_time * 1000, dipoles.conf['vol'][best_idx] * 100 ** 3))
            
            # Best fitted dipoles
            dipole = dipoles[best_idx]
            
            est_diploc=dipole.pos[0]*1000
            
#            print('Act_diploc for %s' %dfname + '= %s' %act_dip[par['nD']-1,:]) 
            print('Est_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %np.around(est_diploc,1))
#            print('Loc_error for %s' %dfname + '= %.1fmm\n' %loc_err)
            print('Dip_amplitude for %s-' %dfname + '%s' %stimcat + '= %.2fnAm' %(dipole.amplitude.max()*10e8))
            
            resultfile = resultfile
            fid = open(resultfile, 'a')
            fid.writelines(dfname + '-chan_' + par['chancat']+ '-' + icaed + '-idx_'+ idx + '-' + stimcat + '-%.3f'%best_time+ '-DipoleFit'
                           '     :  %s' %np.round(est_diploc,2) + '  %.2f' %(dipole.amplitude.max()*10e8) + '  %.2f\n' % 0.00)
            fid.close()
            #fid.writelines('\n')
            
#            dipole.plot_locations(trans, subject=subject, subjects_dir=subjects_dir, mode='orthoview',
#                                  coord_frame='head', idx=idx, show_all=True, ax=None, block=False, 
#                                  show=True, verbose=None) # just change dipole to dipoles to plot all dipoles
#            fontdict={'fontsize': 15, 'fontweight' : 1,
#                      'verticalalignment': 'top', #(u'top', u'bottom', u'center', u'baseline')
#                      'horizontalalignment': 'left'}
#            plt.title('%s ' %stimcat, fontdict=fontdict)
            
#            plt.savefig(out_path + dfname + '_DipoleFit_'+ str(par['numtry'])+ '.png', 
#                                dpi=100, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, 
#                                format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)
            
#% Dipole source signal (Virtual electrode)            
            if par['calc_dip_VE']=='':
                fwdd, stcc= mne.make_forward_dipole(dipoles, bem, evoked.info, trans)
                pred_evoked=mne.simulation.simulate_evoked(fwdd, stcc, evoked.info, cov=None, nave=np.inf)
                
                virt_sig_ev = mne.fit_dipole(evoked_full, data_cov, bem, trans,
                               pos=dipoles.pos[best_idx], ori=dipoles.ori[best_idx])[0]
                virt_sig_orig_ev = mne.fit_dipole(epochs[stimcat].average(), data_cov, bem, trans,
                               pos=dipoles.pos[best_idx], ori=dipoles.ori[best_idx])[0]
                virt_raw=raw_copy.copy().crop(0,50)
                virt_raw.data=virt_raw.get_data()
                virt_raw.comment=evoked.comment
                virt_raw.nave=evoked.nave
                virt_raw._aspect_kind=evoked._aspect_kind
                virt_raw.first=evoked.first
                virt_raw.last=evoked.last
                virt_sig_raw = mne.fit_dipole(virt_raw, data_cov, bem, trans,
                               pos=dipoles.pos[best_idx], ori=dipoles.ori[best_idx])[0]
                
                
                if par['more_plots']=='yes':
                    pred_evoked.comment=stimcat+'-pred_evoked'
                    #pred_evoked.plot(spatial_colors=True, titles='Evoked resp for simulated evoked at dipoles (%s)'%stimcat)
                    #pred_evoked.plot_topo(title='Topoplot for simulated evoked at dipoles (%s)'%stimcat)
                    pred_evoked.plot_topomap(title='Topomap for simulated evoked at dipoles (%s)'%stimcat)
                    #pred_evoked.plot_joint(times='peaks', title='Joint Topoplot for simulated evoked at dipoles (%s)'%stimcat)
                    
                    # rememeber to create a subplot for the colorbar
                    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[10., 3.4])
                    vmin, vmax = -400, 400  # make sure each plot has same colour range
                    
                    # first plot the topography at the time of the best fitting (single) dipole
                    plot_params = dict(times=best_time, ch_type='mag', outlines='skirt', colorbar=False)
                    
                    evoked.plot_topomap(time_format='Measured field', axes=axes[0], **plot_params)
                    
                    # compare this to the predicted field
                    pred_evoked.plot_topomap(time_format='Predicted field', axes=axes[1], **plot_params)
                    
                    # Subtract predicted from measured data (apply equal weights)
                    diff = mne.combine_evoked([evoked, -pred_evoked], weights='equal')
                    plot_params['colorbar'] = True
                    diff.plot_topomap(time_format='Difference', axes=axes[2], **plot_params)
                    plt.suptitle('Comparison of measured and predicted fields for %s ' %stimcat +
                                 'at {:.0f} ms'.format(best_time * 1000.), fontsize=15)
                    
                    #virt_sig_ev.plot(),  plt.suptitle('Virtual evoked data at the best fitted dipole (%s)'%stimcat, fontsize=15)
                    
                    virt_sig_orig_ev.plot(),  plt.suptitle('Virtual original evoked data at the best fitted dipole(%s)'%stimcat, fontsize=15)
                    
                    #virt_sig_raw.plot(),  plt.suptitle('Virtual raw data at the best fitted dipole(%s)'%stimcat, fontsize=15)
            
#            dip_fixedd = mne.fit_dipole(evoked_full, data_cov, bem, trans,
#                           pos=dipole.pos[0], ori=dipole.ori[0])[0]
#            dip_fixedd.plot()
#            
#%% Calculate multiple dipoles >>>>>>>>>>>>> 
    if par['multi_dipfit']=='yes':
        ICappliedondata='yes'
        if ICappliedondata=='yes':
            icaed='ICAed_data_'
        else:
            icaed=''
        
        for stimcat in par['event_dict'].keys()[0:st_len]:
            if 'VEF' in stimcat:
                par['ctrlwin']=[-0.200, -0.050]
                par['actiwin']=[0.050, 0.200]
            elif 'AEF' in stimcat:
                par['ctrlwin']=[-0.150, -0.020]
                par['actiwin']=[0.020, 0.150]
            elif 'SEF' in stimcat:
                par['ctrlwin']=[-0.100, -0.010]
                par['actiwin']=[0.010, 0.100]
            idx='gof' # 'amplitude' ''
            n_dipoles = 5 
        
            evoked=epochs[stimcat].average().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
            data_cov = mne.compute_covariance(epochs[stimcat], tmin=par['actiwin'][0], 
                                              tmax=par['actiwin'][1], method='empirical')   # shrunk
            #t_peak=evoked.get_peak(ch_type='grad')[1] # Taking the peak point from evoked data
                         
            t_points = evoked.times[np.argsort(evoked.data.max(axis=0))[-n_dipoles:]][::-1] # peak time points            
            evdata = np.empty((evoked.data.shape[0],0)) # define an empty array whith shape (306,0)
            for ii in range(0,n_dipoles):
                ev = evoked.copy().crop(t_points[ii], t_points[ii])
                evdata = np.append(evdata, ev.data, axis=1)
            #plt.plot(evdata)

            ev.times = t_points
            ev.data = evdata
            
            mdipole = mne.fit_dipole(ev, data_cov, bem, trans=trans, min_dist=2.5, 
                                          n_jobs=1, pos=None, ori=None, verbose=True)[0]
            est_diplocs=mdipole.pos*1000
            est_amps=mdipole.amplitude*10e9
            est_amps.shape=(n_dipoles,1)
            
            # find the time point with highest GOF to plot
            best_idx = np.argmax(mdipole.gof)
            best_time=mdipole.times[best_idx]
            
            est_diplocs_amps= np.concatenate((est_diplocs, est_amps), axis=1)
            print('Est_diploc & Dip_amplitude for %s-' %dfname + '%s' %stimcat + '= %s' %np.round(est_diplocs_amps))
            
            mdipole.plot_locations(trans, subject=subject, subjects_dir=subjects_dir, mode='orthoview',
                              coord_frame='head', idx=idx, show_all=True, ax=None, block=False, 
                              show=True, verbose=None)
            
            resultfile = resultfile
            fid = open(resultfile, 'a')
            fid.writelines(dfname + '-chan_' + par['chancat']+ '-' + icaed + 'ncov_' + par['nc_meth'] + '-idx_'+ idx + '-' + stimcat + '-DipoleFit'
                           '     :  %s\n' %np.round(est_diplocs_amps))
            fid.close()
            
#%% Dipole source signal (Virtual electrode)            
            if par['calc_dip_VE']=='--':
                fwdd, stcc= mne.make_forward_dipole(mdipole, bem, evoked.info, trans)
                pred_evoked=mne.simulation.simulate_evoked(fwdd, stcc, evoked.info, cov=None, nave=np.inf)
                
                virt_sig_ev = mne.fit_dipole(evoked, data_cov, bem, trans,
                               pos=mdipole.pos[best_idx], ori=mdipole.ori[best_idx])[0]
                virt_sig_orig_ev = mne.fit_dipole(epochs[stimcat].average(), data_cov, bem, trans,
                               pos=mdipole.pos[best_idx], ori=mdipole.ori[best_idx])[0]
                virt_raw=raw_copy.copy().crop(0,50)
                virt_raw.data=virt_raw.get_data()
                virt_raw.comment=evoked.comment
                virt_raw.nave=evoked.nave
                virt_raw._aspect_kind=evoked._aspect_kind
                virt_raw.first=evoked.first
                virt_raw.last=evoked.last
                virt_sig_raw = mne.fit_dipole(virt_raw, data_cov, bem, trans,
                               pos=mdipole.pos[best_idx], ori=mdipole.ori[best_idx])[0]
                
                if par['more_plots']=='yes':
                    pred_evoked.comment=stimcat+'-pred_evoked'
                    #pred_evoked.plot(spatial_colors=True, titles='Evoked resp for simulated evoked at dipoles (%s)'%stimcat)
                    #pred_evoked.plot_topo(title='Topoplot for simulated evoked at dipoles (%s)'%stimcat)
                    pred_evoked.plot_topomap(times=ev.times, title='Topomap for simulated evoked at dipoles (%s)'%stimcat)
                    #pred_evoked.plot_joint(times=ev.times, title='Joint Topoplot for simulated evoked at dipoles (%s)'%stimcat)
                    
                    # rememeber to create a subplot for the colorbar
                    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[10., 3.4])
                    vmin, vmax = -400, 400  # make sure each plot has same colour range
                    
                    # first plot the topography at the time of the best fitting (single) dipole
                    plot_params = dict(times=best_time, ch_type='mag', outlines='skirt', colorbar=False)
                    
                    evoked.plot_topomap(time_format='Measured field', axes=axes[0], **plot_params)
                    
                    # compare this to the predicted field
                    pred_evoked.plot_topomap(time_format='Predicted field', axes=axes[1], **plot_params)
                    
                    # Subtract predicted from measured data (apply equal weights)
                    evoked2=evoked
#                    evoked2.times=t_points
#                    evoked2.data=evoked.data[:,np.argsort(evoked.data.max(axis=0))[-n_dipoles:][::-1]]
#                    
#                    diff = mne.combine_evoked([evoked2, -pred_evoked], weights='equal')
#                    
#                    plot_params['colorbar'] = True
#                    diff.plot_topomap(time_format='Difference', axes=axes[2], **plot_params)
#                    plt.suptitle('Comparison of measured and predicted fields for %s ' %stimcat +
#                                 'at {:.0f} ms'.format(best_time * 1000.), fontsize=15)
                    
                    #virt_sig_ev.plot(),  plt.suptitle('Virtual evoked data at the best fitted dipole (%s)'%stimcat, fontsize=15)
                    
                    virt_sig_orig_ev.plot(),  plt.suptitle('Virtual original evoked data at the best fitted dipole(%s)'%stimcat, fontsize=15)
                    
                    #virt_sig_raw.plot(),  plt.suptitle('Virtual raw data at the best fitted dipole(%s)'%stimcat, fontsize=15)
            
                
#%% Inverse modeling: MNE/dSPM on evoked and raw data
    if par['other_SL']=='n o':
        ICappliedondata='yes'
        if ICappliedondata=='yes':
            icaed='ICAed_data_'
        else:
            icaed=''
            
        for method in ['dSPM', "MNE", "sLORETA"]: # ompute inverse solution for these methods
            
            for stimcat in par['event_dict'].keys()[0:st_len]:
                if 'VEF' in stimcat:
                    par['ctrlwin']=[-0.200, -0.050]
                    par['actiwin']=[0.050, 0.200]
                    snr=3
                elif 'AEF' in stimcat:
                    par['ctrlwin']=[-0.150, -0.020]
                    par['actiwin']=[0.020, 0.150]
                    snr=4
                elif 'SEF' in stimcat:
                    par['ctrlwin']=[-0.100, -0.010]
                    par['actiwin']=[0.010, 0.100]
                    snr=5
            
                #evoked=epochs.average()
                fwd = mne.pick_types_forward(fwd, meg=True, eeg=False) 
                evoked=epochs[stimcat].average().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
                noise_cov = mne.compute_covariance(epochs[stimcat], tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1], method='empirical') # shrunk
                   
                # make an MEG inverse operator
                inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov,       #fixed='auto', limit_depth_chs=True, rank=None, use_cps=None,
                                                                          loose=1.0, depth=0.8, verbose=True)                
                mne.minimum_norm.write_inverse_operator(subjects_dir + subject + '/' + dfname + '-'+ icaed + '%s' %stimcat + '-meg-oct-6-inv.fif',
                                       inverse_operator)
            
                lambda2 = 1. / snr ** 2 # Regularization parameter
                src=inverse_operator['src'] # src=src_v
                stccc=mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, method=method, 
                                                     pick_ori=None, prepared=False, label=None, verbose=True)
                tcrop=0.0
                #####stc.crop(0.020, stc.times[-1])                
                stccc.save(out_path + '/'+ dfname + '_' + icaed + stimcat + '-'+ method + '_volumetric')
        
                img=mne.save_stc_as_volume(out_path + '/' + dfname + '_' + icaed + stimcat + '-' + method + '_inverse.nii.gz', stccc, 
                                           fwd['src'], dest='mri', mri_resolution=False)
                # Alternatively Save it as a nifti file
                # import ninabel as nib
                # nib.save(img, 'mne_%s_inverse.nii.gz' % method)
            
                t_peak=stccc.get_peak()[1]
            
                stccc_peak=stccc.get_peak() # stc data peak timepoint and vertex
            
                thresh = stccc.data.max()*70/100
                timepoint = int(t_peak//stccc.tstep - stccc.times[0]//stccc.tstep)
            
                est_loc=fwd['src'][0]['rr'][stccc_peak[0]]*1000 # in mili meter
                #loc_err=np.sqrt(np.sum(np.square(act_dip[par['nD']-1]-est_loc))); # Calculate loc_err

                #print('Act_diploc for %s' %dfname + '= %s' %act_dip[par['nD']-1,:]) 
                print('Est_diploc for %s-' %dfname + '%s' %stimcat + '= %s' %np.around(est_loc,1))
                #print('Loc_error for %s' %dfname + '= %.1fmm\n' %loc_err)
                print('Peak_value for %s-' %dfname + '%s' %stimcat + '= %.2f\n' % stccc.data.max())
            
                resultfile = resultfile
                fid = open(resultfile, 'a')
                nullgap={'dSPM':'   ', 
                         'MNE' :'    ',
                         'sLORETA':''                            }
                fid.writelines(dfname + '-chan_' + par['chancat']+ '-' + icaed + 'ncov_' + par['nc_meth'] +
                               '-reg_%.2f-' %lambda2 + stimcat + '-' + method + nullgap[method] + ' :  %s' %np.round(est_loc,2) + 
                                '  %.2f' %stccc.data.max() + '  %.2f\n' % 0.00 )
                fid.close()
                
                # Plot brain in 3D with PySurfer if available (for corticaly constrained)
                if par['SL_cc']=='yesss':
                    brain = stccc.plot(hemi='rh', subjects_dir=subjects_dir, initial_time=0.1, time_unit='s')
                    brain.show_view('lateral')
                    brain.save_image('DICS_map.png')
                    
                plot_stat_map(index_img(img, int(tcrop*1000) + timepoint), mrifile, threshold=thresh, 
                              title='%s-'%dfname + '%s'%stimcat + ' /' + method + ' (tpeak=%.3f s.)\n' % stccc.times[int(tcrop*1000) + timepoint] + 
                              'PeakValue= %.3f' % stccc.data.max() + ' / Reg= %.3f\n' % lambda2 + 
                              'Est_loc= %.1f' % est_loc[0]+', %.1f' % est_loc[1]+', %.1f ' % est_loc[2] +
                              '/ Loc_err= %s mm' % '') 
                # plt.pause(1.0)
                # manager=plt.get_current_fig_manager()
                # manager.window.showMaximized()
                
                plt.figure()
                ts_show = -200  # show the 200 largest responses
                plt.plot(1e3 * stccc.times, stccc.data[np.argsort(stccc.data.max(axis=1))[ts_show:]].T)
                plt.title(method +' STC plot for %s-'%dfname + '%s'%stimcat + ' for %d largest sources'%abs(ts_show))
    #            plt.plot(1e3 * stccc.times, stccc.data.T)#[::100, :].T) # Uncomment to plot all stc sources
    #            plt.title('STC plot for %s-'%dfname + '%s'%stimcat)
                plt.xlabel('time (ms)')
                plt.ylabel('%s value' % method + '@Reg= %.3f' %lambda2)
                plt.show()
            
#                fid = open(resultfile, 'a')
#                fid.writelines('\n')
#                fid.close()
#                
#                if par['savefig']=='yes' or par['savefig_res']=='yes': 
#                    plt.savefig(out_path + dfname + '_' + method + '_Source_plot_1_'+ str(par['numtry'])+ '.png', 
#                                dpi=150, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, 
#                                format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None)


#plt.close('all') 



