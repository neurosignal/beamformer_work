#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:02:14 2018 (Using Xfit BEM)

@author: amit jaiswal @ MEGIN (Elekta Oy), Helsinki, Finland

==============================================================================
The script can be used in three modes:
    1..BF for data simulated at any defined location using Elekta SIM_RAW
    2..BF for data simulated at any defined location using MNE python modified 
        function i.e. simulate_stc_at_fixed_location
    3. BF for data simulated at any random location using MNE python function 
        i.e. simulate_stc_sparse
    4. BF for overlaping an evoked phantom data on a resting state data
==============================================================================

"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
#from random import choice
from scipy.io import loadmat
from scipy.spatial import distance
import mne
import subprocess
from os.path import split, splitext, exists
from os import remove, mkdir
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
import sklearn as sk
from scipy.spatial import ConvexHull#, Delaunay
from mne.io.proj import (_read_proj, make_projector, _write_proj, _needs_eeg_average_ref_proj)
from mne.utils import check_fname, logger, verbose, warn
from itertools import product
plt.rcParams.setdefault
plt.rcParams.update({'font.size':15})
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
#warnings.filterwarnings('ignore')
#import psfutils as psf
#import skimage as ski
plt.close('all')
#mlab.close(all=True)
print(__doc__)
#% %##############################################################################
## %gui qt

act_dip=loadmat('/net/bonsai/home/amit/Documents/MATLAB/biomag_phantom.mat') # TRIUX
act_dip=act_dip['biomag_phantom']
par={'mode'             : 3,
    'prep'              : '',                
    'badch'             : [],
    'apply_ica'         : 'no',
    'SLmeths'           : ['lcmv', 'dics', 'sloreta', 'dSPM', 'dfit', 'MNE', 'rap_music'],
    'icaext'            : '-bp_2-95_ICAed',
    'dics'              : '',
    'rap_music'         : '',
    'calc_dip_VE'       : '',
    'do_dipfit'         : '',
    'multi_dipfit'      : '',
    'other_SL'          : 'no', 
    'savefig'           : 'yes',
    'result_plot'       : 'yes',
    'save_resplot'      : 'yes',
    'savefig_res'       : 'yes',
    'maxfilter'         : '',
    'notchfilter'       : '',
    'bandpass'          : '',
    'SL_cort'           : 'no',
    'gridres'           : 5.0,
    'calc_bem'          : 'yes',
    'bem_sol'           : 'inner_skull', #sphere
    'visual'            : 'no',
    'powspect'          : '',
    'check_trial'       : '',
    'browse'            : '',
    'more_plots'        : 'sel',
    'models_plot'       : '',
    'numtry'            : 1,
    'bin'               : 0.1,
    'var_cut'           : [0.001, 98.0],
    'save_ave_afterICA' : 'yes',
    'trialwin'          : [-0.500, 0.500],
    'ctrlwin'           : [-0.500, -0.0],
    'actiwin'           : [0.000, 0.500],
    'bpfreq'            : [2, 45]}

subjects_dir, subject = '/net/qnap/data/rd/ChildBrain/FS_SUBS_DIR/subjects/', 'jukka_nenonen2'
trans = '/net/qnap/data/rd/ChildBrain/Simulation/' + subject + '-trans.fif' # subjects_dir + subject + '/mri/transforms/' + subject + '-trans.fif'
trans = subjects_dir + 'jukka_nenonen/mri/brain-neuromag/sets/jukka_nenonen-amit-121118-MNEicp-trans.fif'
trans = subjects_dir + 'jukka_nenonen/mri/brain-neuromag/sets/jukka_nenonen-amit-131118-MNEicp-trans.fif'
mrifile= subjects_dir + subject + '/mri/T1.mgz'
surffile= subjects_dir + subject + '/bem/watershed/' + subject + '_brain_surface'

# Define local functions >>>>>>>>>
rng = np.random.RandomState(42)
def data_fun(times):
    """Function to generate random source time courses"""
    return (50e-9 * np.sin(30. * times) *
            np.exp(- (times - 0.15 + 0.05 * rng.randn(1)) ** 2 / 0.01))
    
def closest_node(node, nodes): # function to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

def simulate_stc_at_fixed_location(src, n_dipoles, times, loc,
                    data_fun=lambda t: 20*1e-9 * np.sin(2 * np.pi * 20 * t), #lambda t: -20*np.exp(-0.02*t)* np.sin(2*np.pi*20*t/1000)*(t>=0), #
                    labels=None, subject=None, subjects_dir=None):
    src = mne.source_space._ensure_src(src, verbose=True)
    subject_src = src[0].get('subject_his_id')
    if subject is None:
        subject = subject_src
    elif subject_src is not None and subject != subject_src:
        raise ValueError('subject argument (%s) did not match the source '
                         'space subject_his_id (%s)' % (subject, subject_src))
    data = np.zeros((n_dipoles, len(times)))
    for i_dip in range(n_dipoles):
        data[i_dip, :] = data_fun(times)

    if labels is None:
        vss= np.zeros([1,n_dipoles], dtype=int)
        vssloc= np.zeros([n_dipoles,3], dtype=float)
        for ii in range(n_dipoles):
            if loc[ii][0]<=0:
                vssloc[ii,:] = closest_node(loc[ii], src[0]['rr'])[0]
                vss[0,ii]    = closest_node(loc[ii], src[0]['rr'])[1]
            else:
                vssloc[ii,:] = closest_node(loc[ii], src[1]['rr'])[0]
                vss[0,ii]    = closest_node(loc[ii], src[1]['rr'])[1]
        datas = data
        vsss = np.sort(vss).tolist()
    else:
        print('Error: Not included here, see here: ~/anaconda2/lib/python2.7/site-packages/mne/smulation/source.py')
    tmin, tstep = times[0], np.diff(times[:2])[0]
    assert datas.shape == data.shape
    cls = mne.source_estimate.SourceEstimate if len(vss) == 2 else mne.source_estimate.VolSourceEstimate
    stc = cls(datas, vertices=vsss, tmin=tmin, tstep=tstep, subject=subject)
    return stc

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
        mne.viz.plot_alignment(base_raw.info, trans, subject=subject, subjects_dir=subjects_dir, fig=None,
                               surfaces=['head-dense', 'inner_skull'], coord_frame='head', show_axes=True,
                               meg=False, eeg='original', dig=True, ecog=True, bem=None, seeg=True,
                               src=mne.read_source_spaces(src), mri_fiducials=False,  verbose=True) 
    return src

def write_and_read_subject_labels(subjects_dir, subject, fwd_fixed_ori, bem, trans, plot_used_verts, plot_bem, bem2src_mindist,
                                   hemi, ds_f=1, n_extra=1, x_min=5, y_min=5, z_min=1, x_max=1, y_max=5, z_max=5): #% Write subject specific labels, lh,  rh, lh+rh
    surf_h = mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) # Plot bem surface over src
    lhsrc = fwd_fixed_ori['src'][0]
    lhvert = lhsrc['rr'][lhsrc['vertno']]
    rhsrc = fwd_fixed_ori['src'][1]
    rhvert = rhsrc['rr'][rhsrc['vertno']]
    distanc = {}
    sup_inf_vert = {}
    sup_inf_dist = {}
    shorted_inside = {}
    for hem in [hemi]:#['lh', 'rh']:
        distanc[hem] = np.zeros([len(eval(hem + 'src')['rr']),1])
        for ii in range(len(eval(hem + 'src')['rr'])): # This section detects points with min and max dist from center
            if ii in eval(hem + 'src')['vertno']:
                distanc[hem][ii,:] = np.sqrt(np.sum(np.square([0.0,0.0,0.0] - eval(hem + 'src')['rr'][ii,:])))*1000 
            else:
                distanc[hem][ii,:] = 70.0 # just a value between min and max dist for the other cortex
        if n_extra>0:
            sup_inf_vert[hem] =  np.argsort(distanc[hem], axis=0)
            sup_inf_dist[hem] =  np.vstack((distanc[hem][np.argsort(distanc[hem], axis=0)][-n_extra:], 
                                            distanc[hem][np.argsort(distanc[hem], axis=0)][:n_extra]))
        else:
            sup_inf_vert[hem] =  np.array([])
        
        shorted_inside[hem] = {} # This section detects points with min & max on the basis of x,y,z
        for cord in range(3):
            shorted = np.argsort(eval(hem + 'src')['rr'][:,cord])
            shorted_inside[hem][cord] = np.array([], dtype=int)
            for ii in shorted:
                if ii in eval(hem + 'src')['vertno']:
                    shorted_inside[hem][cord] = np.append(shorted_inside[hem][cord],ii)

    for hem in [hemi]:#['lh', 'rh']:
        vert_sel_ = np.unique(np.hstack((eval(hem + 'src')['vertno'][0:len(eval(hem + 'src')['vertno'])//ds_f*ds_f].reshape(len(eval(hem + 'src')['vertno'])//ds_f, ds_f)[:,0], 
                                        sup_inf_vert[hem][:n_extra].reshape(n_extra), sup_inf_vert[hem][-n_extra:].reshape(n_extra),
                                        shorted_inside[hem][0][:x_min], shorted_inside[hem][0][-x_max:], 
                                        shorted_inside[hem][1][:y_min], shorted_inside[hem][1][-y_max:],
                                        shorted_inside[hem][2][:z_min], shorted_inside[hem][2][-z_max:])))
        
        vert_sel=np.empty([0,], 'int')
        for jj in vert_sel_: # this section ommits the vertices which are  
            closest_index, closest_point = closest_vert(eval(hem + 'src')['rr'][jj], surf_h['rr'])
            least_dist_act = np.sqrt(np.sum(np.square(eval(hem + 'src')['rr'][jj]-closest_point)))*1000
            if least_dist_act>bem2src_mindist: # in mm
                vert_sel = np.append(vert_sel, jj)
        print('%s (%d vertices) is downsampled at %d. => Got %d vertices.'%(hem, len(eval(hem + 'vert')), ds_f, len(eval(hem + 'vert'))//ds_f)) 
        print('In %s, %d nearest & %d furthest extra vertices were selected => Got %d vertices.'%(hem, n_extra, n_extra, 2*n_extra))
        print('In %s, %d extra vertices were selected in each of the minimum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_in, 2*3*n_xyz_in))
        print('In %s, %d extra vertices were selected in each of the maximum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_out, 2*3*n_xyz_out))
        print('Total #vertices = %d(selected)-%d(within %.1fmm close to the bem surface)=%d. \nFinally got %d vertices.\n'%(len(vert_sel_), (len(vert_sel_)-len(vert_sel)), bem2src_mindist, len(vert_sel), len(vert_sel)))
        
        label_file = '%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel), hem)
        fid = open(label_file, 'w')
        fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
        fid.writelines('%d \n' %len(vert_sel))
        for ii in vert_sel:
            fid.writelines('%d ' %ii)
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(eval(hem + 'src')['rr'][ii]*1000))
        fid.close()
        
    cmb_vertno = np.concatenate((lhsrc['vertno'], rhsrc['vertno']))
    cmb_vertno = np.sort(cmb_vertno, axis=-1, kind='quicksort')
    label_file = '%s%s/%s_%dpts_lh+rh.label'%(subjects_dir, subject, subject, len(cmb_vertno))
    fid = open(label_file, 'w')
    fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
    fid.writelines('%d \n' %len(cmb_vertno))
    for ii in cmb_vertno:
        fid.writelines('%d ' %ii)
        if ii in lhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(lhsrc['rr'][ii]*1000))
        elif ii in rhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(rhsrc['rr'][ii]*1000))
    fid.close()
    label = mne.read_label('%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel), hemi)) 
    label = list([label])
    if plot_used_verts=='yes':  
        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
        mlab.points3d(label[0].pos[:,0], label[0].pos[:,1], label[0].pos[:,2], mode='sphere', scale_factor=0.005, color=(1,0,0))
        mlab.points3d(lhvert[:,0], lhvert[:,1], lhvert[:,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
        mlab.points3d(rhvert[:,0], rhvert[:,1], rhvert[:,2], mode='sphere', scale_factor=0.00125/2, color=(0.8,0.9,0.8))
        mlab.points3d(0.0,0.0,0.0, mode='sphere', scale_factor=0.005, color=(0,0,1))
        if plot_bem=='yes':  
            mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0.8,0.9,0.8))
        ##mlab.show()
    print(label)
    return label
#% %
def write_and_read_subject_labels2(subjects_dir, subject, fwd_fixed_ori, bem, trans, plot_used_verts, plot_bem, bem2src_mindist,
                                   hemi, ds_res=40, n_extra=1, x_min=5, y_min=5, z_min=1, x_max=1, y_max=5, z_max=5): #% Write subject specific labels, lh,  rh, lh+rh
    surf_h = mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) # Plot bem surface over src
    lhsrc = fwd_fixed_ori['src'][0]
    lhvert = lhsrc['rr'][lhsrc['vertno']]
    rhsrc = fwd_fixed_ori['src'][1]
    rhvert = rhsrc['rr'][rhsrc['vertno']]
    distanc = {}
    sup_inf_vert = {}
    sup_inf_dist = {}
    shorted_inside = {}
    for hem in [hemi]:#['lh', 'rh']:
        distanc[hem] = np.zeros([len(eval(hem + 'src')['rr']),1])
        for ii in range(len(eval(hem + 'src')['rr'])): # This section detects points with min and max dist from center
            if ii in eval(hem + 'src')['vertno']:
                distanc[hem][ii,:] = np.sqrt(np.sum(np.square([0.0,0.0,0.0] - eval(hem + 'src')['rr'][ii,:])))*1000 
            else:
                distanc[hem][ii,:] = 70.0 # just a value between min and max dist for the other cortex
        if n_extra>0:
            sup_inf_vert[hem] =  np.argsort(distanc[hem], axis=0)
            sup_inf_dist[hem] =  np.vstack((distanc[hem][np.argsort(distanc[hem], axis=0)][-n_extra:], 
                                            distanc[hem][np.argsort(distanc[hem], axis=0)][:n_extra]))
        else:
            sup_inf_vert[hem] =  np.array([])
        
        shorted_inside[hem] = {} # This section detects points with min & max on the basis of x,y,z
        for cord in range(3):
            shorted = np.argsort(eval(hem + 'src')['rr'][:,cord])
            shorted_inside[hem][cord] = np.array([], dtype=int)
            for ii in shorted:
                if ii in eval(hem + 'src')['vertno']:
                    shorted_inside[hem][cord] = np.append(shorted_inside[hem][cord],ii)
        
        ds_vertnum = np.empty([0,], 'int')
        ds_verts   = np.empty([0,3], 'int')
        total_vert = deepcopy(lhvert)
        cntt = -1
        for jj in eval(hem + 'src')['vertno']:
            cntt = cntt + 1
            if total_vert[cntt,0]!=[1000.0] and total_vert[cntt,1]!=[1000.0] and total_vert[cntt,2]!=[1000.0]:
                ds_vertnum = np.append(ds_vertnum, jj)
                ds_verts   = np.vstack((ds_verts, total_vert[cntt]))
                temp_vert  = deepcopy(total_vert[cntt,:])
                #total_vert[cntt,:] = [0.0,0.0,0.0]
                temp_dist=0.0
                while temp_dist<ds_res:
                    closest_index, closest_point = closest_vert(temp_vert, total_vert)
                    temp_dist = np.sqrt(np.sum(np.square(temp_vert-closest_point)))*1000
                    total_vert[closest_index,:]=[1000.0,1000.0,1000.0]
        del total_vert
#        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
#        mlab.points3d(lhvert[:,0], lhvert[:,1], lhvert[:,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
#        mlab.points3d(ds_verts[:,0], ds_verts[:,1], ds_verts[:,2], mode='sphere', scale_factor=0.005, color=(1,0,0))

    for hem in [hemi]:#['lh', 'rh']:
        vert_sel_ = np.unique(np.hstack((ds_vertnum, 
                                        sup_inf_vert[hem][:n_extra].reshape(n_extra), sup_inf_vert[hem][-n_extra:].reshape(n_extra),
                                        shorted_inside[hem][0][:x_min], shorted_inside[hem][0][-x_max:], 
                                        shorted_inside[hem][1][:y_min], shorted_inside[hem][1][-y_max:],
                                        shorted_inside[hem][2][:z_min], shorted_inside[hem][2][-z_max:])))
        
        vert_sel=np.empty([0,], 'int')
        for jj in vert_sel_: # this section ommits the vertices which are  
            closest_index, closest_point = closest_vert(eval(hem + 'src')['rr'][jj], surf_h['rr'])
            least_dist_act = np.sqrt(np.sum(np.square(eval(hem + 'src')['rr'][jj]-closest_point)))*1000
            # print(jj, least_dist_act)
            if least_dist_act>bem2src_mindist: # in mm
                vert_sel = np.append(vert_sel, jj)
#        print('%s (%d vertices) is downsampled at %d. => Got %d vertices.'%(hem, len(eval(hem + 'vert')), ds_f, len(eval(hem + 'vert'))//ds_f)) 
#        print('In %s, %d nearest & %d furthest extra vertices were selected => Got %d vertices.'%(hem, n_extra, n_extra, 2*n_extra))
#        print('In %s, %d extra vertices were selected in each of the minimum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_in, 2*3*n_xyz_in))
#        print('In %s, %d extra vertices were selected in each of the maximum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_out, 2*3*n_xyz_out))
#        print('Total #vertices = %d(selected)-%d(within %.1fmm close to the bem surface)=%d. \nFinally got %d vertices.\n'%(len(vert_sel_), (len(vert_sel_)-len(vert_sel)), bem2src_mindist, len(vert_sel), len(vert_sel)))
        
        label_file = '%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel), hem)
        fid = open(label_file, 'w')
        fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
        fid.writelines('%d \n' %len(vert_sel))
        for ii in vert_sel:
            fid.writelines('%d ' %ii)
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(eval(hem + 'src')['rr'][ii]*1000))
        fid.close()
        
    cmb_vertno = np.concatenate((lhsrc['vertno'], rhsrc['vertno']))
    cmb_vertno = np.sort(cmb_vertno, axis=-1, kind='quicksort')
    label_file = '%s%s/%s_%dpts_lh+rh.label'%(subjects_dir, subject, subject, len(cmb_vertno))
    fid = open(label_file, 'w')
    fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
    fid.writelines('%d \n' %len(cmb_vertno))
    for ii in cmb_vertno:
        fid.writelines('%d ' %ii)
        if ii in lhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(lhsrc['rr'][ii]*1000))
        elif ii in rhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(rhsrc['rr'][ii]*1000))
    fid.close()
    label = mne.read_label('%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel), hemi)) 
    label = list([label])
    if plot_used_verts=='yes':  
        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
        mlab.points3d(label[0].pos[:,0], label[0].pos[:,1], label[0].pos[:,2], mode='sphere', scale_factor=0.005, color=(1,0,0))
        mlab.points3d(lhvert[:,0], lhvert[:,1], lhvert[:,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
        if hem=='lh':
            for ii in range(len(lhvert)):
                iii = lhsrc['vertno'][ii]
                print(ii, iii)
                mlab.points3d(lhvert[ii,0], lhvert[ii,1], lhvert[ii,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
                mlab.text3d(lhvert[ii,0], lhvert[ii,1], lhvert[ii,2], ' ' + str(iii), scale=0.0005, color=(0,1,0))
            mlab.roll(-90)
            a = anim() # Starts the animation.
            ##mlab.show()
                
        mlab.points3d(rhvert[:,0], rhvert[:,1], rhvert[:,2], mode='sphere', scale_factor=0.00125/2, color=(0.8,0.9,0.8))
        mlab.points3d(0.0,0.0,0.0, mode='sphere', scale_factor=0.005, color=(0,0,1))
        if plot_bem=='yes':  
            mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0.8,0.9,0.8))
        ##mlab.show()
    print(label)
    return label
#% %
def write_and_read_subject_labels3(subjects_dir, subject, fwd_fixed_ori, bem, trans, plot_used_verts, plot_bem, bem2src_mindist,
                                   hemi, ds_res=40, n_extra=1, x_min=5, y_min=5, z_min=1, x_max=1, y_max=5, z_max=5): #% Write subject specific labels, lh,  rh, lh+rh
    surf_h = mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) # Plot bem surface over src
    lhsrc = fwd_fixed_ori['src'][0]
    lhvert = lhsrc['rr'][lhsrc['vertno']]
    rhsrc = fwd_fixed_ori['src'][1]
    rhvert = rhsrc['rr'][rhsrc['vertno']]
    distanc = {}
    sup_inf_vert = {}
    sup_inf_dist = {}
    shorted_inside = {}
    vert_sel_verts = {}
#    vert_sel_final = {}
    for hem in [hemi]:#['lh', 'rh']:
        #>>>>Using bem2src_mindist:: this section ommits the vertices which are very close to bem 
        vert_sel=np.empty([0,], 'int')
        for jj in eval(hem + 'src')['vertno']: 
            closest_index, closest_point = closest_vert(eval(hem + 'src')['rr'][jj], surf_h['rr'])
            least_dist_act = np.sqrt(np.sum(np.square(eval(hem + 'src')['rr'][jj]-closest_point)))*1000
            # print(jj, least_dist_act)
            if least_dist_act>bem2src_mindist: # in mm
                vert_sel = np.append(vert_sel, jj)
        vert_sel_verts[hem] = lhsrc['rr'][vert_sel]
        print('In %s, (%d vertices) were reduced to %d after removing the vertices very close to BEM surface.=> Got %d vertices.'%(hem, len(eval(hem + 'vert')), len(vert_sel), len(vert_sel))) 
        
        #>>>>Using n_extra:: this section calculate distance of all vertices in vert_sel_verts[hem] from origin for selecting min. & max. distant vertices
        distanc[hem] = np.zeros([len(eval(hem + 'src')['rr']),1])
        for ii in range(len(eval(hem + 'src')['rr'])): # This section detects points with min and max dist from center
            if ii in vert_sel:
                distanc[hem][ii,:] = np.sqrt(np.sum(np.square([0.0,0.0,0.0] - eval(hem + 'src')['rr'][ii,:])))*1000 
            else:
                distanc[hem][ii,:] = 70.0 # just a value between min and max dist for the other cortex
        if n_extra>0:
            sup_inf_vert[hem] =  np.argsort(distanc[hem], axis=0)
            sup_inf_dist[hem] =  np.vstack((distanc[hem][np.argsort(distanc[hem], axis=0)][-n_extra:], 
                                            distanc[hem][np.argsort(distanc[hem], axis=0)][:n_extra]))
            print('\nIn %s, %d extra vertices at min(%.1fmm) & max(%.1fmm) distance from centre were selected.\n'%(hem, 2*n_extra, sup_inf_dist[hem][1], sup_inf_dist[hem][0]))
        else:
            sup_inf_vert[hem] =  np.array([])
            
        
        #>>>>Using min/max X/Y/Z:: this section shorts vert_sel on the basis of x,y,z coorinates to detects points with min & max X,Y,Z
        shorted_inside[hem] = {} 
        shorted_inside[hem][0] = np.array([], dtype=int)
        shorted_inside[hem][1] = np.array([], dtype=int)
        shorted_inside[hem][2] = np.array([], dtype=int)
        for cord in range(3):
            shorted = np.argsort(vert_sel_verts[hem][:,cord])
            shorted_inside[hem][cord] = np.argsort(vert_sel_verts[hem][:,cord])
        
        shorted_inside[hem] = {} # detecting points with min & max on the basis of x,y,z coords values
        for cord in range(3):
            shorted = np.argsort(eval(hem + 'src')['rr'][:,cord])
            shorted_inside[hem][cord] = np.array([], dtype=int)
            for ii in shorted:
                if ii in vert_sel: #eval(hem + 'src')['vertno']:
                    shorted_inside[hem][cord] = np.append(shorted_inside[hem][cord],ii)
                    
        #>>>>Using ds_res:: this section provides an equally distributed fewer vertices which are ds_res mm apart from nearly vertex
        ds_vertnum = np.empty([0,], 'int')
        ds_verts   = np.empty([0,3], 'int')
        total_vert = deepcopy(vert_sel_verts[hem])
        cntt = -1
        for jj in vert_sel:
            cntt = cntt + 1
            if total_vert[cntt,0]!=[1000.0] and total_vert[cntt,1]!=[1000.0] and total_vert[cntt,2]!=[1000.0]:
                ds_vertnum = np.append(ds_vertnum, jj)
                ds_verts   = np.vstack((ds_verts, total_vert[cntt]))
                temp_vert  = deepcopy(total_vert[cntt,:])
                #total_vert[cntt,:] = [0.0,0.0,0.0]
                temp_dist=0.0
                while temp_dist<ds_res:
                    closest_index, closest_point = closest_vert(temp_vert, total_vert)
                    temp_dist = np.sqrt(np.sum(np.square(temp_vert-closest_point)))*1000
                    total_vert[closest_index,:]=[1000.0,1000.0,1000.0]
        del total_vert

    for hem in [hemi]:#['lh', 'rh']:
#        vert_sel1_ = ds_vertnum
#        vert_sel2_ = sup_inf_vert[hem][:n_extra].reshape(n_extra), sup_inf_vert[hem][-n_extra:].reshape(n_extra)
#        vert_sel3_ = shorted_inside[hem][0][:x_min], shorted_inside[hem][0][-x_max:] 
#        vert_sel4_ = shorted_inside[hem][1][:y_min], shorted_inside[hem][1][-y_max:]
#        vert_sel5_ = shorted_inside[hem][2][:z_min], shorted_inside[hem][2][-z_max:]
#        vert_sel_ = np.unique(np.hstack((vert_sel1_, vert_sel2_, vert_sel3_, vert_sel4_, vert_sel5_)))
        vert_sel_ = np.unique(np.hstack((ds_vertnum, 
                                        sup_inf_vert[hem][:n_extra].reshape(n_extra), sup_inf_vert[hem][-n_extra:].reshape(n_extra),
                                        shorted_inside[hem][0][:x_min], shorted_inside[hem][0][-x_max:], 
                                        shorted_inside[hem][1][:y_min], shorted_inside[hem][1][-y_max:],
                                        shorted_inside[hem][2][:z_min], shorted_inside[hem][2][-z_max:])))
        #vert_sel_final[hem] = eval(hem + 'src')['rr'][vert_sel_,:]
        
#        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
#        mlab.points3d(lhvert[:,0], lhvert[:,1], lhvert[:,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
#        mlab.points3d(rhvert[:,0], rhvert[:,1], rhvert[:,2], mode='sphere', scale_factor=0.00125/2, color=(0.9,0.8,0.9))
#        mlab.points3d(vert_sel_verts[hem][:,0], vert_sel_verts[hem][:,1], vert_sel_verts[hem][:,2], mode='sphere', 
#                      scale_factor=0.001250, color=(0.0,1.0,0.0), opacity=0.5)
#        mlab.points3d(ds_verts[:,0], ds_verts[:,1], ds_verts[:,2], mode='sphere', scale_factor=0.005, color=(1,0,0))
#        mlab.points3d(vert_sel_final[hem][:,0], vert_sel_final[hem][:,1], vert_sel_final[hem][:,2], mode='sphere', 
#                      scale_factor=0.00650, color=(0.0,0.0,1.0), opacity=0.4)
#        
#        
#        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
#        mlab.points3d(eval(hem + 'src')['rr'][vert_sel1_,0], eval(hem + 'src')['rr'][vert_sel1_,1], eval(hem + 'src')['rr'][vert_sel1_,2], mode='2dcross', scale_factor=0.0125, color=(1,0,1))
#        mlab.points3d(eval(hem + 'src')['rr'][vert_sel2_,0], eval(hem + 'src')['rr'][vert_sel2_,1], eval(hem + 'src')['rr'][vert_sel2_,2], mode='2dcircle', scale_factor=0.0125, color=(0,0,1))
#        mlab.points3d(eval(hem + 'src')['rr'][vert_sel3_,0], eval(hem + 'src')['rr'][vert_sel3_,1], eval(hem + 'src')['rr'][vert_sel3_,2], mode='2dsquare', scale_factor=0.0125, color=(1,0,0))
#        mlab.points3d(eval(hem + 'src')['rr'][vert_sel4_,0], eval(hem + 'src')['rr'][vert_sel4_,1], eval(hem + 'src')['rr'][vert_sel4_,2], mode='2dtriangle', scale_factor=0.0125, color=(1,0,1))
#        mlab.points3d(eval(hem + 'src')['rr'][vert_sel5_,0], eval(hem + 'src')['rr'][vert_sel5_,1], eval(hem + 'src')['rr'][vert_sel5_,2], mode='2ddiamond', scale_factor=0.0125, color=(0,0,0))
        
        
#        vert_sel=np.empty([0,], 'int')
#        for jj in vert_sel_: # this section ommits the vertices which are  
#            closest_index, closest_point = closest_vert(eval(hem + 'src')['rr'][jj], surf_h['rr'])
#            least_dist_act = np.sqrt(np.sum(np.square(eval(hem + 'src')['rr'][jj]-closest_point)))*1000
#            # print(jj, least_dist_act)
#            if least_dist_act>bem2src_mindist: # in mm
#                vert_sel = np.append(vert_sel, jj)
#        print('%s (%d vertices) is downsampled at %d. => Got %d vertices.'%(hem, len(eval(hem + 'vert')), ds_f, len(eval(hem + 'vert'))//ds_f)) 
#        print('In %s, %d nearest & %d furthest extra vertices were selected => Got %d vertices.'%(hem, n_extra, n_extra, 2*n_extra))
#        print('In %s, %d extra vertices were selected in each of the minimum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_in, 2*3*n_xyz_in))
#        print('In %s, %d extra vertices were selected in each of the maximum of X, Y and Z => Got %d vertices.'%(hem, n_xyz_out, 2*3*n_xyz_out))
#        print('Total #vertices = %d(selected)-%d(within %.1fmm close to the bem surface)=%d. \nFinally got %d vertices.\n'%(len(vert_sel_), (len(vert_sel_)-len(vert_sel)), bem2src_mindist, len(vert_sel), len(vert_sel)))
        
        label_file = '%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel_), hem)
        fid = open(label_file, 'w')
        fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
        fid.writelines('%d \n' %len(vert_sel_))
        for ii in vert_sel_:
            fid.writelines('%d ' %ii)
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(eval(hem + 'src')['rr'][ii]*1000))
        fid.close()
    cmb_vertno = np.concatenate((lhsrc['vertno'], rhsrc['vertno']))
    cmb_vertno = np.sort(cmb_vertno, axis=-1, kind='quicksort')
    label_file = '%s%s/%s_%dpts_lh+rh.label'%(subjects_dir, subject, subject, len(cmb_vertno))
    fid = open(label_file, 'w')
    fid.writelines('%s \n' %'# Label from subject surface src in head coordinate_frame')
    fid.writelines('%d \n' %len(cmb_vertno))
    for ii in cmb_vertno:
        fid.writelines('%d ' %ii)
        if ii in lhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(lhsrc['rr'][ii]*1000))
        elif ii in rhsrc['vertno']:
            fid.writelines('%.2f %.2f %.2f 0\n' %tuple(rhsrc['rr'][ii]*1000))
    fid.close()
    label = mne.read_label('%s%s/%s_%dpts_%s.label'%(subjects_dir, subject, subject, len(vert_sel_), hemi)) 
    label = list([label])
    if plot_used_verts=='yes':  
        mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
        mlab.points3d(label[0].pos[:,0], label[0].pos[:,1], label[0].pos[:,2], mode='sphere', scale_factor=0.005, color=(1,0,0))
        for iii in range(len(label[0].pos)):
            mlab.text3d(label[0].pos[iii,0], label[0].pos[iii,1], label[0].pos[iii,2], ' '+str(iii+1), scale=0.005, color=(1.0,0,0))
        mlab.points3d(lhvert[:,0], lhvert[:,1], lhvert[:,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
#        if hem=='lh':
#            for ii in range(len(lhvert)):
#                iii = lhsrc['vertno'][ii]
#                print(ii, iii)
#                mlab.points3d(lhvert[ii,0], lhvert[ii,1], lhvert[ii,2], mode='sphere', scale_factor=0.00125, color=(0.9,0.8,0.9))
#                mlab.text3d(lhvert[ii,0], lhvert[ii,1], lhvert[ii,2], ' ' + str(iii), scale=0.0005, color=(0,1,0))
#            mlab.roll(-90)
#            a = anim() # Starts the animation.
#            ##mlab.show()
        mlab.points3d(vert_sel_verts[hem][:,0], vert_sel_verts[hem][:,1], vert_sel_verts[hem][:,2], mode='sphere', 
                      scale_factor=0.001250, color=(0.0,1.0,0.0), opacity=0.5)        
        mlab.points3d(rhvert[:,0], rhvert[:,1], rhvert[:,2], mode='sphere', scale_factor=0.00125/2, color=(0.8,0.9,0.8))
        mlab.points3d(0.0,0.0,0.0, mode='sphere', scale_factor=0.005, color=(0,0,1))
        if plot_bem=='yes':  
            mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0.8,0.9,0.8))
        print('##mlab.show() commented, otherwise getting stucked')
    print(label)
    return label

#% %
def stlVolume(points,triangles):
    pnts = points.T
    tris = triangles.T
    d13= np.vstack((tuple(pnts[0,tris[1,:]]-pnts[0,tris[2,:]]), tuple(pnts[1,tris[1,:]]-pnts[1,tris[2,:]]), tuple(pnts[2,tris[1,:]]-pnts[2,tris[2,:]])))
    d12= np.vstack((tuple(pnts[0,tris[0,:]]-pnts[0,tris[1,:]]), tuple(pnts[1,tris[0,:]]-pnts[1,tris[1,:]]), tuple(pnts[2,tris[0,:]]-pnts[2,tris[1,:]])))
    cr = np.cross(d13.T, d12.T).T
    area = 0.5 * np.sqrt(cr[0,:] ** 2 + cr[1,:] ** 2 + cr[2,:]**2) # Area of each triangle
    totalArea = sum(area)
    crNorm = (np.sqrt(cr[0,:] ** 2 + cr[1,:] ** 2 + cr[2,:] ** 2)).T
    zMean = ((pnts[2,tris[0,:]]+ pnts[2,tris[1,:]] + pnts[2,tris[2,:]])/3).T               
    
    nz = -cr[2,:]/crNorm # z component of normal for each triangle
    volume = area*zMean*nz # contribution of each triangle
    totalVolume = sum(volume) # divergence theorem
    return(totalVolume*1000*1000*1000, totalArea*1000*1000)

def _check_reference(inst):
    """Check for EEG ref."""
    if _needs_eeg_average_ref_proj(inst.info):
        raise ValueError('EEG average reference is mandatory for inverse '
                         'modeling, use set_eeg_reference method.')
    if inst.info['custom_ref_applied']:
        raise ValueError('Custom EEG reference is not allowed for inverse '
                         'modeling.')
def _check_ch_names(inv, info):
    """Check that channels in inverse operator are measurements."""
    inv_ch_names = inv['eigen_fields']['col_names']

    if inv['noise_cov'].ch_names != inv_ch_names:
        raise ValueError('Channels in inverse operator eigen fields do not '
                         'match noise covariance channels.')
    data_ch_names = info['ch_names']

    missing_ch_names = sorted(set(inv_ch_names) - set(data_ch_names))
    n_missing = len(missing_ch_names)
    if n_missing > 0:
        raise ValueError('%d channels in inverse operator ' % n_missing +
                         'are not present in the data (%s)' % missing_ch_names)

def _pick_channels_inverse_operator(ch_names, inv):
    """Return data channel indices to be used knowing an inverse operator.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = list()
    for name in inv['noise_cov'].ch_names:
        try:
            sel.append(ch_names.index(name))
        except ValueError:
            raise ValueError('The inverse operator was computed with '
                             'channel %s which is not present in '
                             'the data. You should compute a new inverse '
                             'operator restricted to the good data '
                             'channels.' % name)
    return sel

def estimate_3_snr(evoked_pst, evoked_pre, inverse_operator, fwd, src_amp, megchan):
    #% % MNE python implementation
#    evoked = epochs.average()
    #evoked1=evoked.copy()
#    evoked1=evoked_pst#evoked.copy().crop(0.01,0.100)
    # inv=inverse_operator
    from scipy.stats import chi2
    _check_reference(evoked_pst)
    _check_ch_names(inverse_operator, evoked_pst.info)
    inv = mne.minimum_norm.prepare_inverse_operator(inverse_operator, evoked_pst.nave, 1. / 9., 'MNE')
    sel = _pick_channels_inverse_operator(evoked_pst.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
#                    evoked1.plot(spatial_colors=True, gfp=True, titles='Original evoked data', time_unit='ms')
    data_white = np.dot(inv['whitener'], np.dot(inv['proj'], evoked_pst.data[sel]))
#                    plt.figure(), plt.plot(evoked1.data[sel].T), plt.title('evoked: data only')
#                    plt.figure(), plt.imshow(inv['proj']), plt.title('inv[''proj'']')
#                    plt.figure(), plt.plot(np.dot(inv['proj'], evoked1.data[sel]).T), plt.title('np.dot(inv[''proj''], evoked1.data[sel]).T')
#                    plt.figure(), plt.imshow(inv['whitener']), plt.title('inv[''whitener'']')
#                    plt.figure(), plt.plot(data_white.T), plt.title('whitened data')
    
    data_white_ef = np.dot(inv['eigen_fields']['data'], data_white)
#                    plt.figure(), plt.imshow(inv['eigen_fields']['data']), plt.title('inv[''eigen_fields''][''data'']')
#                    plt.figure(), plt.plot(data_white_ef.T), plt.title('eigen field whitened data')
    n_ch, n_times = data_white.shape
    
    # Adapted from mne_analyze/regularization.c, compute_regularization
    n_zero = (inv['noise_cov']['eig'] <= 0).sum()
    logger.info('Effective nchan = %d - %d = %d'
                % (n_ch, n_zero, n_ch - n_zero))
    signal = np.sum(data_white ** 2, axis=0)  # sum of squares across channels
#                    plt.figure(), plt.plot((data_white ** 2).T)
#                    plt.figure(), plt.plot(np.sum(data_white ** 2, axis=0))
    noise = n_ch - n_zero
    snr = signal / noise
    # plt.figure(), plt.plot(snr)
    
    # Adapted from noise_regularization
    lambda2_est = np.empty(n_times)
    lambda2_est.fill(10.)
    remaining = np.ones(n_times, bool)
    
    # deal with low SNRs
    bad = (snr <= 1)
    lambda2_est[bad] = 100.
    remaining[bad] = False
    
    # parameters
    lambda_mult = 0.9
    sing2 = (inv['sing'] * inv['sing'])[:, np.newaxis]
    val = chi2.isf(1e-3, n_ch - 1)
    for n_iter in range(1000):
        # get_mne_weights (ew=error_weights)
        # (split newaxis creation here for old numpy)
        f = sing2 / (sing2 + lambda2_est[np.newaxis][:, remaining])
        f[inv['sing'] == 0] = 0
        ew = data_white_ef[:, remaining] * (1.0 - f)
        # check condition
        err = np.sum(ew * ew, axis=0)
        remaining[np.where(remaining)[0][err < val]] = False
        if not remaining.any():
            break
        lambda2_est[remaining] *= lambda_mult
    else:
        warn('SNR estimation did not converge')
    snr_est = 1.0 / np.sqrt(lambda2_est)
    snr = np.sqrt(snr)
    #return snr, snr_est
#                    plt.figure(), plt.plot(snr)
#                    #plt.figure(), 
#                    plt.plot(snr_est)
    peak_ch, peak_time = evoked_pst.get_peak(ch_type='mag' if not megchan=='grad' else 'grad')
    tp = int((peak_time - par['actiwin'][0])*evoked.info['sfreq'])
    snr_mne = snr[tp]
    snr_est_mne = snr_est[tp]
        
#    snr_comp_file = '/net/qnap/data/rd/ChildBrain/BeamComp/Result_master/snr_comp_file.csv'
#    fid = open(snr_comp_file, 'a+')
#    if prepcat==maxf[0] and dipnum==dipoles[0] and subcat==amps[0]:
#        fid.writelines('\n%s\n,'   %comments)
#    fid.writelines('%s,'     %dfname)
#    fid.writelines('%s,'     %subcat)
#    fid.writelines('%d,'    %dipnum)
#    fid.writelines('%.2f,'   %SNR)
#    fid.writelines('%.2f,'   %SNR_EST)
#                    fid.writelines('\n')
#                    fid.close()

#% % using 10log10(s/sigma_b)  ## mentioned in NBRC paper 
#    evoked_pre = epochs.average().crop(par['ctrlwin'][0], par['ctrlwin'][1])
#                    plt.figure(), plt.plot(evoked_pre.data.T), plt.title('original baseline')
    inv_bs = mne.minimum_norm.prepare_inverse_operator(inverse_operator, evoked_pre.nave, 1. / 9., 'MNE')
    sel = _pick_channels_inverse_operator(evoked_pre.ch_names, inv_bs)
    logger.info('Picked %d channels from the data' % len(sel))
    baseline_white = np.dot(inv_bs['whitener'], np.dot(inv_bs['proj'], evoked_pre.data[sel]))
#                    plt.figure(), plt.plot(baseline_white.T), plt.title('whitened baseline')
    #sigma_b0 = np.std(evoked_pre.data, axis=0)
    sigma_b1 = np.std(baseline_white.T)#, axis=1)
#                    plt.figure(), plt.plot(data_white.T), plt.title('whitened data')
    p2p = (np.diff([np.min(data_white), np.max(data_white)]))[0]
    SNR_physio = 10*np.log10(p2p/sigma_b1)
#                    fid = open(snr_comp_file, 'a+')
#    fid.writelines('%.2f,'   %SNRphy)
#                    fid.writelines('\n')
#                    fid.close()

#% % SNRdB = 10log10(a2/N*sum(bk2/sk2)  by Goldenholz paper
#    kk=fwd['sol']['data']
#    kkk=np.dot(kk, kk.T)
#                    plt.figure(), plt.imshow(kk), plt.title('leadfield matrix from fwd')
#                    plt.figure(), plt.imshow(kk.T), plt.title('transpose of leadfield matrix from fwd')
#                    plt.figure(), plt.imshow(kkk), plt.title('dot product of leadfield and its transpose')
    
    #SNRdB = 10*np.log10((1/fwd['nchan'])*np.sum((data_white ** 2 / ), axis=0))
    #a=1000*1e-9
    data_act = evoked_pst.data
    data_bl  = evoked_pre.data
    a = 1
#    SNR_golden_ = 10*np.log10((np.square(a)/fwd['nchan']) * np.array([(data_white[t,:]**2 / np.var(baseline_white[t,:])) for t in range(n_zero,fwd['nchan'])]).sum())
    SNR_golden = 10*np.log10((np.square(a)/fwd['nchan']) * np.array([(data_act[t,:]**2 / np.var(data_bl[t,:])) for t in range(n_zero,fwd['nchan'])]).sum())
    SNR_golden2= 10*np.log10((np.square(src_amp)/fwd['nchan']) * np.array([(data_act[t,:]**2 / np.var(data_bl[t,:])) for t in range(n_zero,fwd['nchan'])]).sum())
    SNR_golden3= 10*np.log10((np.square(src_amp*1e-9)/fwd['nchan']) * np.array([(data_act[t,:]**2 / np.var(data_bl[t,:])) for t in range(n_zero,fwd['nchan'])]).sum())
#    fid.writelines('%.2f,'   %SNRdB)
#    fid.writelines('\n')
#    fid.close()
#    
#    plt.close('all')

    return snr_mne, snr_est_mne, SNR_physio, SNR_golden, SNR_golden2, SNR_golden3

def closest_vert(node, nodes): # class to find closest points
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index, nodes[closest_index]

@mlab.animate(delay=50, ui=True)
def anim():
  f = mlab.gcf()
  while 1:
      f.scene.camera.azimuth(10)
      f.scene.render()
      yield

#%%
if par['mode']==1: # Using Evoked data from Xfit dipole simulation
    loc  = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40], [50, 50, 50]] #mm
    amps = (10, 20, 40, 50, 80, 100, 150, 200, 300, 400, 500, 750, 1000) #nAm
    tangle, rangle, noise = [44, 86, 88, 89, 90], [68, 95, 96, 96, 96], [0,0]
    for dip in range(5):
        for amp in amps:
            #% %
            #base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw.fif'
            if par['maxf']=='raw':
                base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw.fif'
            elif par['maxf']=='sss':
                base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw_sss.fif'

            data_path = out_path = '/net/qnap/data/rd/ChildBrain/Simulation/'
            resultfile = '/net/bonsai/home/amit/git/ChildBrain/BeamComp/BeamComp_Resultfiles/Simulations/' + 'Simulations_Est_SourceLoc.csv'
            #mne.io.read_raw_fif(base_file, preload=True).filter(None, 95.0).plot()
            name_ext = tuple([loc[dip][0], loc[dip][1], loc[dip][2], tangle[dip], rangle[dip], noise[0], noise[1], amp])
            evoked_file = data_path + 'Xfit_%d_%d_%d_%d_%d_noise_%d_%d_%dnA_sim_evo2.fif'%name_ext # Q = -20*exp(-0.02*t)*sin(2*pi*10*t/1000)*(t>=0)
            #mne.read_evokeds(evoked_file)[0].plot(spatial_colors=True, gfp=True)
            evoked_temp = mne.read_evokeds(evoked_file)[0]
            evoked_temp = evoked_temp.copy().crop(-0.0, evoked_temp.times[-1])
            evoked_post_file = evoked_file[:-4] + '-post.fif'
            evoked_temp.save(evoked_post_file)
            simraw_file = '%s/%s_OVER_%s.fif' %(split(splitext(evoked_file)[0])[0], split(splitext(evoked_file)[0])[1], split(splitext(base_file)[0])[1]) 
            dfname = split(splitext(simraw_file)[0])[1]
            
            sim_cmd = '/neuro/bin/util/sim_raw -i %s -o %s -e %s 1.0 1.2 1 1 101 -a 5' %(base_file, simraw_file, evoked_post_file)   #+ ' > ' + out_path + par['bf_type'] + '/run_log.txt'
            print(subprocess.check_output(sim_cmd, shell=True))
            print('Completed successfully.........\n')
            
            raw = mne.io.read_raw_fif(simraw_file, preload=True)
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            # proj = mne.read_proj(simraw_file)
            # raw.info['projs'] += proj
            # raw.filter(None, 45.0).plot(events) 
            # mne.viz.plot_events(events)
            
            raw.pick_types(meg=True)
            raw.info['bads'] = ['MEG2233', 'MEG0111']
            raw.drop_channels(raw.info['bads'])
            
            raw.notch_filter(50, filter_length='auto', phase='zero') # Notch filter 
            raw.notch_filter(25, filter_length='auto', phase='zero') # Notch filter 
            raw.filter(par['bpfreq'][0], par['bpfreq'][1], l_trans_bandwidth=0.1,#min(max(2 * 0.01, 2), 2), 
                       h_trans_bandwidth=min(max(70 * 0.01, 2.), raw.info['sfreq'] / 2. - 70), 
                       filter_length='auto', phase='zero')
#            raw.plot(events)
#            raw.plot_psd(average=False, spatial_colors=True, line_alpha=0.5, fmin=0.0, fmax=100.0)        

elif par['mode']==2 or par['mode']==3: # Using Evoked data from MNE simulation
    #% Load real data as templates
    #data_path = mne.datasets.sample.data_path()
    #
    #raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
    #events = mne.find_events(raw)
    #proj = mne.read_proj(data_path + '/MEG/sample/sample_audvis_ecg-proj.fif')
    #raw.info['projs'] += proj
    #raw.info['bads'] = ['MEG 2443', 'EEG 053']  # mark bad channels
    #
    #fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
    #ave_fname = data_path + '/MEG/sample/sample_audvis-no-filter-ave.fif'
    #cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
    #
    #fwd = mne.read_forward_solution(fwd_fname)
    #fwd = mne.pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    #cov = mne.read_cov(cov_fname)
    #info = mne.io.read_info(ave_fname)
    
    #label_names = ['Aud-lh', 'Aud-rh']
#    labels = [mne.read_label(mne.datasets.sample.data_path() + '/''MEG/sample/labels/%s.label' % ln)
#              for ln in label_names]
    #%
    #base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw.fif'
    base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw_sss.fif' # replaced fron raw to sss on 29 Nov
    base_raw = mne.io.read_raw_fif(base_file, preload=True,verbose=True) 
    #proj = mne.read_proj(base_file)
    #base_raw.info['projs'] += proj
    #base_raw.info['bads'] = ['MEG2233']  # mark bad channels
    #dfname=split(splitext(base_file)[0])[1]
    data_path = out_path = '/net/qnap/data/rd/ChildBrain/Simulation/NEW_25_seglabBEM6/'#NEW_DS10/' #
    if not exists(data_path):
        mkdir(data_path)
    temp_events = mne.find_events(mne.io.read_raw_fif(mne.datasets.sample.data_path() + '/MEG/sample/sample_audvis_raw.fif'))
    base_events = temp_events[temp_events[:,0]<=base_raw.times[-1]*base_raw.info['sfreq']]
    base_events[:,2] = 5 # change all trigger value to 5
    
    base_raw.notch_filter(np.arange(25,76,25), filter_length='auto', phase='zero')## added on 29 Nov
    base_raw.filter(1.0,95.0, filter_length='auto', phase='zero')# added on 29 Nov
#    base_raw.plot(base_events, proj=False)
    base_epochs = mne.Epochs(base_raw.copy().pick_types(meg=True), base_events, 5, par['trialwin'][0], par['trialwin'][1],  
                            #baseline=(par['trialwin'][0], 0), 
                            picks=None, preload=True,  
                            flat=None, proj=False, decim=1,reject_tmin=None, reject_tmax=None, detrend= None, 
                            on_missing='error', reject_by_annotation=True, verbose=True)
    
    base_cov = mne.compute_covariance(base_epochs, method='shrinkage') # compute noise cov
#    base_cov.plot(base_epochs.info, show_svd=False, proj=True)
    base_evoked = base_epochs.average()
#    base_evoked.plot(spatial_colors=True)
    info = base_evoked.info

elif par['mode']==4: # Using phantom Evoked data
    evoked = mne.read_evokeds('') 

src = define_source_space_and_plot(mode='surface', plot_alignment='yes', gridres=5.0, 
                                   spacing='ico4', surf_type='pial', mindist=5.0, exclude=10.0)
#mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
#                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0.8,0.9,0.8))
##src_ = mne.read_source_spaces(src, verbose=True)
#mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(800, 700))
#mlab.points3d(src_[0]['rr'][:,0], src_[0]['rr'][:,1], src_[0]['rr'][:,2], mode='sphere', scale_factor=0.0002, color=(0.5,0.1,0))
#mlab.points3d(src_[1]['rr'][:,0], src_[1]['rr'][:,1], src_[1]['rr'][:,2], mode='sphere', scale_factor=0.0002, color=(0,0.1,0.5))

#model = mne.make_bem_model(subject=subject, ico=5, conductivity=(0.33,), subjects_dir=subjects_dir, verbose=True)
#bem_ = mne.make_bem_solution(model) # bem-solution
#
#surf_h_ = mne.transform_surface_to(bem_.copy()['surfs'][0], 'head', mne.read_trans(trans)) 
#mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700))
#mlab.triangular_mesh(surf_h_['rr'][:,0], surf_h_['rr'][:,1], surf_h_['rr'][:,2], surf_h_['tris'], representation='wireframe', 
#                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0,1,0))
#mlab.points3d(bem_['surfs'][0]['rr'][:,0], bem_['surfs'][0]['rr'][:,1], bem_['surfs'][0]['rr'][:,2], mode='sphere', scale_factor=0.0005, color=(1,0,1))
###mlab.show()

#bem2=mne.read_bem_solution('/neuro/databases/bem/jukka_nenonen-3_Layer-bem-sol.fif', verbose=True)

#bem = mne.read_bem_solution('/neuro/databases/bem/JNe_for_simulation4-bem-sol.fif', verbose=True)
#bem = mne.read_bem_solution('/neuro/databases/bem/JNe_Simu_1L_033_m_MRI_MNEicptrans_smooth-bem-sol.fif', verbose=True)
#bem = mne.read_bem_solution('/neuro/databases/bem/JNe_Simu_1L_033_m_MRI_MNEicptrans_smooth4-bem-sol.fif', verbose=True)
#bem = mne.read_bem_solution('/neuro/databases/bem/JNe_simu_1L_m_33_MNEtrans-bem-sol.fif', verbose=True)
bem = mne.read_bem_solution('/neuro/databases/bem/JNe_Simu_1L_033_m_MRI_MNEicptrans_smooth5-bem-sol.fif', verbose=True)
print('BEM coordinate frame ID: ' + str(bem['surfs'][0]['coord_frame'][0]))
#nanin = bem.copy()['surfs'][0]['rr'] # to remove the nan from bem
#nanindx = np.unique(np.argwhere(np.isnan(nanin))[:,0])
#for nanindx_ in nanindx:
#    nanin[nanindx_,:] = nanin[nanindx_+1,:]
#bem['surfs'][0]['rr'] = nanin
#del nanin
surf_h = mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) 
surf_simu_bem = surf_h.copy()

if par['more_plots']=='yes':
    mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700)) #>>>>>>>
    mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                         mode='sphere', opacity=0.5, scale_factor=.001, color=(1,1,0))
    #mlab.points3d(bem['surfs'][0]['rr'][:,0], bem['surfs'][0]['rr'][:,1], bem['surfs'][0]['rr'][:,2],mode='sphere', scale_factor=0.0005, color=(1,0,0))
    digs =base_raw.info['dig']
    digss = np.empty([0,3], 'float32')
    for ndig in range(len(base_raw.info['dig'])):
        digss=np.vstack((digss, digs[ndig]['r']))
    mlab.points3d(digss[:,0], digss[:,1], digss[:,2],mode='sphere', scale_factor=0.005, color=(0,0.7,0))
    ###mlab.show()
    
#mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='pial', 
#                                 src=src, slices=range(73,193,5), orientation='sagittal')
trans_val = mne.read_trans(trans, return_all=True)
fwd = mne.make_forward_solution(base_epochs.info, trans=trans, src=src, bem=bem, 
                    meg=True, eeg=False, mindist=2.5, n_jobs=2)#n_jobs=2
#plt.figure(), plt.imshow(np.dot(fwd['sol']['data'], fwd['sol']['data'].T))
# To simulate the data, we need a version of the forward solution where each
# source has a "fixed" orientation, i.e. pointing orthogonally to the surface
# of the cortex.
fwd_fixed_ori = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True, verbose=True)
print("Leadfield size : %d sensors x %d dipoles" % fwd_fixed_ori['sol']['data'].shape)
#plt.figure(), plt.imshow(np.dot(fwd_fixed_ori['sol']['data'], fwd_fixed_ori['sol']['data'].T))
if not 'labelss' in locals():  # ds_f=10 for 263 dipoles
    labelss = write_and_read_subject_labels3(subjects_dir, subject, fwd_fixed_ori, bem, trans, 
                                                plot_used_verts='yes', plot_bem='no', bem2src_mindist=7.5, 
                                                hemi='lh', ds_res=42.0, n_extra=1, x_min=1, y_min=1, z_min=1,
                                                x_max=1, y_max=1, z_max=1) # notice x_max/x_min for lh and rh 

print('%d points in %s hemisphere is being used for data simulation'%(labelss[0].pos.shape[0],'left' if labelss[0].hemi=='lh' else 'right'))
bem2srcdistt = np.empty([0,], 'float') 
for lb in range(len(labelss[0].pos)):
    closest_index, closest_point = closest_vert(labelss[0].pos[lb,:], surf_h['rr'])
    bem2label_dist = np.sqrt(np.sum(np.square(labelss[0].pos[lb,:]-closest_point)))*1000
    bem2srcdistt = np.append(bem2srcdistt, bem2label_dist)
    print('%d\t%s\t%.1f'%(lb, str(labelss[0].pos[lb,:]*1000), bem2label_dist))
labelss_pos, labelss_vertices = labelss[0].pos, labelss[0].vertices
del (bem, src, fwd, surf_h)

#%%
megchans  = ['True','grad', 'mag'][:1]
preprocs  = ['noSSP','raw', 'sss'][:1]#, 'ICA_rmEOG_raw','ICA_rmEOGECG_raw']'tsss'
cov_meths = ['empirical']#, 'shrinkage', 'ledoit_wolf', 'oas', 'shrunk']
weight_norms = ['nai']#, 'unit-noise-gain']
for megchan, preproc, cov_meth, weight_norm in product(megchans, preprocs, cov_meths, weight_norms):##
    amps = [10,30,80,200,300,450,600,800]#
    for amp in amps:#
        for vertnum in range(labelss[0].pos.shape[0]):
            print(preproc,cov_meth, weight_norm, amp, vertnum)
            print('megchan= %s\t preproc= %s\t amp= %d\t vertnum= %d'%(megchan, preproc, amp, vertnum))
            #% %
            if par['mode']==2: #%% Using simulate_stc_at_fixed_location  
                n_dipoles = 1
                dipnum    = 19
                loc = act_dip[dipnum-1:dipnum-1 + n_dipoles]/1000
                #times = np.arange(300, dtype=np.float) / base_raw.info['sfreq'] - 0.1 # fix the time
                times = base_evoked.copy().crop(0.0, 0.2).times#base_evoked.times[-1]).times
                stc = simulate_stc_at_fixed_location(fwd_fixed_ori['src'], n_dipoles, times, loc, #data_fun=data_fun,
                                        labels=None, subject=subject, subjects_dir=subjects_dir)
                
                stc_loc = fwd_fixed_ori['src'][0 if loc[0][0]<=0 else 1]['rr'][int(stc.vertices)]*1000 
        
                plt.figure()#('Simulated stc data for %d dipoles'%n_dipoles) 
                line=plt.plot(stc.times, stc.data.T)
                plt.legend(line[:n_dipoles], map(str, list(np.asarray(range(n_dipoles))+1)), 
                           loc='upper right', fontsize=None, shadow=False, framealpha=0.5)
                print('\nStimulated dipole/stc location is [%.1f, %.1f, %.1f]mm'%tuple(stc_loc))
    
            if par['mode']==3:#%% Using mne.simulation.simulate_sparse_stc
                n_dipoles = 1
                times = base_evoked.copy().crop(0.0, 0.2).times # base_evoked.times[-1]).times
                #labels = write_and_read_subject_labels(subjects_dir, subject, fwd_fixed_ori, plot_used_verts='no', hemi=choice(['lh', 'rh']), ds_f=1)
                if 'label' in locals():
                    del label 
                label = deepcopy(labelss)
                label[0].pos = labelss_pos[vertnum,:] # take only one vertex at a time
                label[0].vertices = labelss_vertices[vertnum]
                
                data_fn = lambda t: (amp/100) * 1e-7 * np.sin(2*10 * np.pi * t)
                stc = mne.simulation.simulate_sparse_stc(fwd_fixed_ori['src'], n_dipoles, times, 
                                    data_fun=data_fn, #lambda t: 20*np.exp(-t)* np.sin(2*np.pi*10*t),
                                    labels=label, random_state=None, location='random', 
                                    subject=subject, subjects_dir=subjects_dir, surf='sphere')
                print(label[0].pos)
                print(fwd_fixed_ori['src'][0]['rr'][int(stc.vertices[0])])
                print(stc.vertices)
                
    #            a,b,c,d,e = zip(*((map(float, line.split()) for line in open('/opt/freesurfer/subjects/jukka_nenonen2/jukka_nenonen2_lh.label'))))
    #            for line in open('/opt/freesurfer/subjects/jukka_nenonen2/jukka_nenonen2_lh.label'):
    #                print(line)
    #            plt.figure()
    #            line = plt.plot(stc.times, stc.data.T)
    #            plt.legend(line[:n_dipoles], map(str, list(np.asarray(range(n_dipoles))+1)), 
    #                       loc='upper right', fontsize=None, shadow=False, framealpha=0.5)
                if len(fwd_fixed_ori['src'])==2:
                    act_src_vert = int(stc.vertices[1 if stc.vertices[0].tolist()==[] else 0])#[1 if lab==None else 0])
                else:
                    act_src_vert = int(stc.vertices)
                
                stc_loc = fwd_fixed_ori['src'][1 if stc.vertices[0].tolist()==[] else 0]['rr'][act_src_vert]*1000 
                # stc_loc = fwd_fixed_ori['src'][0]['rr'][act_src_vert]*1000
                print('\nStimulated dipole/stc location is [%.1f, %.1f, %.1f]mm'%tuple(stc_loc))
                print((True, ': Correct location') if np.around(stc_loc,2).tolist() in label[0].pos*1000 else (False, ': Incorrect location'))
                print('Distance from origin = %.3fmm'%np.sqrt(np.sum(np.square([0,0,0]-stc_loc))))
                
    #            if np.around(stc_loc,2).tolist() in label[0].pos*1000:
    #                kk[vertnum]=1
    #            else:
    #                kk[vertnum]=act_src_vert
            
            if par['mode']==2 or par['mode']==3:# Generate noisy evoked data
                picks = mne.pick_types(base_raw.info, meg=True, exclude='bads')
                iir_filter = mne.time_frequency.fit_iir_model_raw(base_raw, order=5, picks=picks, 
                                                                  tmin=60, tmax=180, verbose=True)[1] # try to understand
                nave = np.Inf #100 # simulate average of 100 epochs
        #        if len(fwd_fixed_ori['src'])!=2:
        #            fwd_fixed_ori['source_ori'] = 1 #faking the value: FIX it
                evoked_file=out_path + 'Simulated_evoked_' + str(amp) + 'nAm_at_%.1f_%.1f_%.1fmm-ave.fif'%tuple(stc_loc)
                if not exists(evoked_file):
                    evoked = mne.simulation.simulate_evoked(fwd_fixed_ori, stc, base_evoked.info, base_cov, nave=nave, use_cps=True,
                                         iir_filter=None, verbose=True)
                    evoked.comment='Simulated @[%.1f,%.1f,%.1f]mm'%tuple(stc_loc)
                    #### evoked.add_proj=projs
                    if len(fwd_fixed_ori['src'])==2 and par['more_plots']=='yes': # Plot
                        mne.viz.plot_sparse_source_estimates(fwd_fixed_ori['src'], stc, bgcolor=(1, 1, 1),
                                                 opacity=0.5, high_resolution=True, verbose=True)
                        plt.figure(),  plt.psd(evoked.data[0])
                    if not nave==np.Inf:
                        evoked=evoked.filter(None, 45.0)
                    evoked.save(evoked_file)
                    # evoked.plot(spatial_colors=True, gfp=True, time_unit='ms')
                    # evoked.plot_topo()
            # #%% generate raw using mne (fix it using: https://users.aalto.fi/~vanvlm1/conpy/)
        #    infoo = mne.create_info(fwd_fixed_ori['info']['ch_names'], raw.info['sfreq'])
        #    infoo.update(fwd_fixed_ori['info'])
        #    simraw_data =  mne.apply_forward_raw(fwd_fixed_ori, stc, infoo, verbose=True)
            
            #% % Generate raw data (FIX it)
            #raw_sim = mne.simulation.simulate_raw(raw, stc3, trans, src, bem2, cov='simple',ecg=False, 
            #                       blink=False, iir_filter=[0.2, -0.2, 0.04], n_jobs=1, verbose=True)
            #%  % Part to overlap the generated evoked over base_file using sim_raw tool
            resultfile = '/net/bonsai/home/amit/git/ChildBrain/BeamComp/BeamComp_Resultfiles/Simulations/' + 'Simulations_Est_SourceLoc_1.csv'
            #mne.io.read_raw_fif(base_file, preload=True).filter(None, 95.0).plot()
            evoked_file = evoked_file
            ##mne.read_evokeds(evoked_file)[0].plot(spatial_colors=True, gfp=True)
            #evoked_temp = mne.read_evokeds(evoked_file)[0]
            #evoked_temp = evoked_temp.copy().crop(-0.0, evoked_temp.times[-1])
            #evoked_post_file = evoked_file[:-4] + '-post.fif'
            #evoked_temp.save(evoked_post_file)
            if preproc=='raw':
                base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw.fif'
            elif preproc=='noSSP':
                base_file = '/net/qnap/data/rd/ChildBrain/Simulation/spont_eo_raw_nossp.fif'
            elif preproc=='sss':
                base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw_sss.fif'
            elif preproc=='tsss':
                base_file = '/net/qnap/data/rd/multimodal/nenonen_jukka/101007/spont_eo_raw_tsss.fif'
            elif preproc=='ICA_rmEOG_raw':
                base_file = '/net/qnap/data/rd/ChildBrain/Simulation/nenonen_jukka_101007_spont_eo_ICA_rmEOG_raw.fif'
            elif preproc=='ICA_rmEOGECG_raw':
                base_file = '/net/qnap/data/rd/ChildBrain/Simulation/nenonen_jukka_101007_spont_eo_ICA_rmEOGECG_raw.fif'
            
            simraw_file = '%s/%s_OVER_%s.fif' %(split(splitext(evoked_file)[0])[0], split(splitext(evoked_file)[0])[1], split(splitext(base_file)[0])[1]) 
# Dectivated on 29 Nov #sim_cmd = '/neuro/bin/util/sim_raw -i %s -o %s -e %s 1.0 1.2 1 1 101 -a 5' %(base_file, simraw_file, evoked_file)   #+ ' > ' + out_path + par['bf_type'] + '/run_log.txt'
            sim_cmd = '/neuro/bin/util/sim_raw -i %s -o %s -e %s 1.1 1.1 1 1 101 -t 60.0 -a 5' %(base_file, simraw_file, evoked_file)   #+ ' > ' + out_path + par['bf_type'] + '/run_log.txt'
            if not exists(simraw_file):
                print(subprocess.check_output(sim_cmd, shell=True))
                print('Simulated raw data file generated successfully.........\n')
            
    #        fname_maxf = simraw_file[:-4] + '_sss.fif'
    #        maxf_cmd = '/neuro/bin/util/maxfilter -f %s -o %s_sss.fif -bad 2233 -autobad on -badlimit 7'%(simraw_file, simraw_file[:-4]) 
    #        maxf_cmd2= '/neuro/bin/util/maxfilter -f %s -o %s_tsss.fif -origin 1.7 15.7 42.9 -frame head -bad 2233 -badlimit 7 -ctc /neuro/databases/ctc/ct_sparse_tkk.fif -cal /neuro/databases/sss/sss_cal_tkk.dat -st 30 -corr 0.980 -force'%(simraw_file, simraw_file[:-4]) #+ ' > ' + out_path + par['bf_type'] + '/run_log.txt'
    #        if not exists(fname_maxf):
    #            print(subprocess.check_output(maxf_cmd, shell=True))
    #            print('MaxFiltered successfully.........\n')
            #%
            raw    = mne.io.read_raw_fif(simraw_file, preload=True) #fname_maxf# 
            events = mne.find_events(raw, stim_channel='STI101', min_duration=0.003)
            dfname = split(splitext(raw.filenames[0])[0])[1]
            #        proj = mne.read_proj(simraw_file)
            #        raw.info['projs'] += proj
            # raw.filter(None, 45.0).plot(events) 
            #mne.viz.plot_events(events)
        #    raw.fix_mag_coil_types()
        #    raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.04), int_order=8, ext_order=3, 
        #            calibration=None, cross_talk=None, st_duration=None, st_correlation=0.98, coord_frame='head',
        #            destination=None, regularize='in', ignore_ref=False, bad_condition='error', head_pos=None,
        #            st_fixed=True, st_only=False, mag_scale=100.0, verbose=None)
            bads=[]
            if not raw.info['projs']==[] or not 'sss' in raw.filenames[0]:
                bads = ['MEG2233'] #raw.info['bads']
                bads = ['MEG2233', 'MEG2212','MEG2332']
                bads += ['MEG0111'] # flat
                #bads += ['MEG0933'] #jump at 96.2s
                raw.drop_channels(bads)
                
            if megchan=='True':
                megchan=bool(megchan)
            raw.pick_types(meg=megchan)#True)
            
            if par['bpfreq'][1]>=46:
                raw.notch_filter(np.arange(25.0,51.0,25.0), filter_length='auto', phase='zero') # Notch filter 
#            elif par['bpfreq'][1]>=23:
#                raw.notch_filter(25.0, filter_length='auto', phase='zero') # Notch filter 
    #        raw.filter(par['bpfreq'][0], par['bpfreq'][1], l_trans_bandwidth=0.1,#min(max(2 * 0.01, 2), 2), 
    #                   h_trans_bandwidth=min(max(70 * 0.01, 2.), raw.info['sfreq'] / 2. - 70), 
    #                   filter_length='auto', phase='zero') 
            raw.filter(par['bpfreq'][0], par['bpfreq'][1], filter_length='auto', phase='zero')
    
        #    raw.filter(par['bpfreq'][0], par['bpfreq'][1], picks=None, filter_length='auto', l_trans_bandwidth='auto', 
        #            h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, phase='zero', 
        #            fir_window='hamming', fir_design='firwin', 
        #            skip_by_annotation=('edge', 'bad_acq_skip'), pad='reflect_limited', verbose=True)
             
        #    raw.plot(events)
            raw.plot_psd(average=False, spatial_colors=True, line_alpha=0.5, fmin=0.0, fmax=100.0) 
            plt.savefig(out_path + dfname + '_raw_plot_psd.png', facecolor='w', edgecolor='w', 
                                        orientation='landscape', bbox_inches='tight', pad_inches=0.2)
        #    remove(simraw_file)
        #    remove(evoked_file)
            #% % Apply ICA 
#            n_components, method, decim, random_state = 25, 'fastica', 3, 23
#            reject = dict(mag=5e-12, grad=4000e-13) 
#            ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state)
#            print(ica)
#            
#            ica.fit(raw, picks=None, start=None, stop=None, decim=decim, reject=reject, 
#                    reject_by_annotation=True, flat=None, verbose=True)
#            print(ica)
#            
#            ica.plot_components(picks=range(0,n_components), ch_type=None, res=64, layout=None, vmin=None, vmax=None, 
#                            cmap='RdBu_r', sensors=True, colorbar=False, title=None, show=True, 
#                            outlines='head', contours=8, image_interp='bilinear', head_pos=None, inst=None)  # can you spot some potential bad guys?-------------
#            ica.plot_sources(raw)
#            
#            ica.apply(raw)
#            print('ICA applied (%s components removed)' %ica.exclude)
#            #raw.save(data_path + dfname + '-bp_'+ str(int(par['bpfreq'][0]))+ '-'+ str(int(par['bpfreq'][1])) + '_ICAed.fif', overwrite=True)
            #% %
##>>commented on 29 Nov>> raw.info.normalize_proj() #>>>>>>>>>>>>>>>>>>>>>>
            reject = dict(mag=4e-12, grad=4000e-13) 
            eventID = int(sim_cmd[-1])
            epochs = mne.Epochs(raw, events, eventID, -0.2, 
                                0.2, baseline=(None,0), picks=None, #reject=reject, 
                                preload=True, flat=None, proj=False, decim=1,
                                reject_tmin=None, reject_tmax=None, detrend= None, 
                                on_missing='error', reject_by_annotation=True, verbose=True)
            badtrls = [14,102,103,106]
            
            par['ctrlwin'] = [-0.100, -0.010]
            par['actiwin'] = [0.010, 0.100]
#            noise_epochs = epochs.copy().crop(tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1])
#            data_epochs  = epochs.copy().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
#            
            # Find trial variance && z-score outliers and index them
            trl_var = np.empty((0,1), 'float')
            #trl_zscore = np.empty((0,1), 'float')
            trlindx = np.arange(0,len(epochs))
            for trnum in range(len(epochs)):
                trl_var= np.vstack((trl_var, max(np.var(np.squeeze(epochs[trnum].get_data()), axis=1))))
            lim1 = (trl_var < np.percentile(trl_var, par['var_cut'][0], interpolation='midpoint')).flatten()
            lim2 = (trl_var > np.percentile(trl_var, par['var_cut'][1], interpolation='midpoint')).flatten()
            outlr_idx = trlindx[lim1].tolist() + trlindx[lim2].tolist()
            plt.figure(), plt.scatter(trlindx, trl_var, marker='D'), plt.ylabel('Max. Variance accros channels-->')
            plt.scatter(outlr_idx, trl_var[outlr_idx],marker='^'), plt.xlabel('Trial number-->')
            plt.scatter(badtrls, trl_var[badtrls],marker='^')
            plt.ylim(min(trl_var), max(trl_var)), plt.title('        Max. variance distribution')          
            bad_trials = np.union1d(badtrls, outlr_idx)
            epochs.drop(bad_trials, reason='eye_blink and high variance', verbose=True) # added on 29 nov
            
#            # Find trial variance && z-score outliers and index them
#            trl_var = np.empty((0,1), 'float')
#            #trl_zscore = np.empty((0,1), 'float')
#            trlindx = np.arange(0,len(epochs))
#            for trnum in range(len(data_epochs)):
#                trl_var= np.vstack((trl_var, max(np.var(np.squeeze(data_epochs[trnum].get_data()), axis=1))))
#            lim1 = (trl_var < np.percentile(trl_var, par['var_cut'][0], interpolation='midpoint')).flatten()
#            lim2 = (trl_var > np.percentile(trl_var, par['var_cut'][1], interpolation='midpoint')).flatten()
#            outlr_idx = trlindx[lim1].tolist() + trlindx[lim2].tolist()
#            plt.figure(), plt.scatter(trlindx, trl_var, marker='D'), plt.ylabel('Max. Variance accros channels-->')
#            plt.scatter(outlr_idx, trl_var[outlr_idx],marker='^'), plt.xlabel('Trial number-->')
#            plt.scatter(badtrls, trl_var[badtrls],marker='^')
#            plt.ylim(min(trl_var), max(trl_var)), plt.title('        Max. variance distribution')          
#            bad_trials = np.union1d(badtrls, outlr_idx)
#            data_epochs.drop(bad_trials, reason='eye_blink and high variance', verbose=True) # added on 29 nov

#            plt.figure()
#            for ntr in range(len(epochs)):
#                plt.clf()
#                plt.plot(epochs[ntr].average().data.T)
#                plt.title(str(ntr))
#                plt.pause(0.1)
            #del raw
            if par['check_trial']=='yes':
                epochs.plot(picks=None, scalings=None, n_epochs=10, n_channels=30, event_colors=None, 
                            title='Epochs plot (cascaded)', events=None, show=True, block=False)
            # epochs.plot_drop_log() 
            # epochs.save(fname[0:-4] + '_%s-%sHz_mne_epo.fif' %tuple(par['bpfreq'])) #(data_path + dfname + '-epo.fif')
            # epochs=mne.read_epochs(data_path + dfname + '-epo.fif', proj=True, preload=True, verbose=True)
            
            evoked = epochs.average()
            evoked.save(simraw_file[:-4] + '-' + str(megchan) + '-ave.fif')
            #plt.close('all')
            #  %%
            evoked.comment=dfname
            if not exists(out_path + dfname + '_evokedplot.png'):
                evoked.plot(spatial_colors=True, gfp=True, time_unit='ms')
                plt.savefig(out_path + dfname + '_evokedplot.png', facecolor='w', edgecolor='w', 
                                        orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                plt.close()
            if not exists(out_path + dfname + '_topoplot.png'):
                evoked.plot_topo()
                plt.savefig(out_path + dfname + '_topoplot.png', facecolor='w', edgecolor='w', 
                                        orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                plt.close()
            #% %
              # Define SNR and reg value (fix it with only gradiometer)
    #        varpst = np.var(evoked_pst.data, axis=0)
    #        ch_idx = np.where(varpst>=max(varpst)*0.75)
    #        ch_name = np.array(evoked_pst.ch_names)[ch_idx[0]]
    #        SNR2 = snr(evoked_pst.data[ch_idx,:].T, evokedpre.avg(ch_idx,:)');
    #        reg2 = eval('SNR2^4/500');
    #        % par.reg = eval(par.reg_form);
    
            #% %
        #    evoked.plot_joint(times='peaks')
            if not 'bem' in locals():
                model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.33,), 
                                               subjects_dir=subjects_dir, verbose=True)
                bem = mne.make_bem_solution(model) # bem-solution
                # mne.write_bem_surfaces('/neuro/databases/bem/' + subject + '-mne_1L_ico4-bem.fif', model)
                # mne.write_bem_solution('/neuro/databases/bem/' + subject + '-mne_1L_ico4-bem-sol.fif', bem)
            #% % Forward solution..................>>
            mode='surface' if par['SL_cort']=='yes' else 'volume'
            if not 'src' in locals():
                src = define_source_space_and_plot(mode, plot_alignment='yes', mindist=2.5, exclude=10.0)
            if not 'fwd' in locals():
                fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                            meg=megchan, eeg=False, mindist=2.5, n_jobs=1)#n_jobs=2
            nchan_fwd = fwd['sol']['data'].shape[0]
            if len(epochs.ch_names)!=nchan_fwd:
                fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, 
                            meg=True, eeg=False, mindist=2.5, n_jobs=1)#n_jobs=2        
            print("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)
            #    mne.write_forward_solution(subjects_dir + subject + '/' + dfname + '_volume-' + 
            #                               str(par['gridres'])+ 'mm_fwd.fif', fwd, overwrite=True)
            surf_h=mne.transform_surface_to(bem.copy()['surfs'][0], 'head', mne.read_trans(trans)) # transform 
            if par['more_plots']=='yes':
                plt.figure(), plt.imshow(np.dot(fwd['sol']['data'], fwd['sol']['data'].T))
                mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700))
                mlab.triangular_mesh(surf_h['rr'][:,0], surf_h['rr'][:,1], surf_h['rr'][:,2], surf_h['tris'], representation='wireframe', 
                                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0,1,0))
                mlab.triangular_mesh(surf_simu_bem['rr'][:,0], surf_simu_bem['rr'][:,1], surf_simu_bem['rr'][:,2], surf_simu_bem['tris'], representation='wireframe', 
                                                     mode='sphere', opacity=0.5, scale_factor=.001, color=(0,1,1))
                mlab.points3d(bem['surfs'][0]['rr'][:,0], bem['surfs'][0]['rr'][:,1], bem['surfs'][0]['rr'][:,2], mode='sphere', scale_factor=0.0005, color=(1,0,1))
                ##mlab.show()
            closest_index, closest_point1 = closest_vert(label[0].pos, surf_h['rr'])
            dist_act2scansurf = np.sqrt(np.sum(np.square(label[0].pos-closest_point1)))*1000
            closest_index, closest_point2 = closest_vert(label[0].pos, surf_simu_bem['rr'])
            dist_act2simusurf = np.sqrt(np.sum(np.square(label[0].pos-closest_point2)))*1000
    
            #% % Apply LCMV
#            par['ctrlwin'] = [-0.100, -0.010]
#            par['actiwin'] = [0.010, 0.100]
            ##cov_meth = 'shrinkage'# 'auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf', 'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage'
            evoked_pst = evoked.copy().crop(tmin=par['actiwin'][0], tmax=par['actiwin'][1])
            evoked_pre = evoked.copy().crop(tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1])
            #evoked_pre.plot(spatial_colors=True, gfp=True, time_unit='ms')
            #evoked_pst.plot(spatial_colors=True, gfp=True, time_unit='ms')
            ##evoked_pre = evoked.copy().crop(tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1])
            #                        evoked = evoked_pst
#            noise_cov = mne.compute_covariance(noise_epochs, method=cov_meth) 
#            data_cov  = mne.compute_covariance(data_epochs, method=cov_meth)   
            noise_cov = mne.compute_covariance(epochs, tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1], method=cov_meth) 
            data_cov  = mne.compute_covariance(epochs, tmin=par['actiwin'][0], tmax=par['actiwin'][1], method=cov_meth)   
            noise_cov.save('%s_%s_%s_noise_cov.fif'%(simraw_file[:-4],str(megchan),cov_meth))
            data_cov.save('%s_%s_%s_data_cov.fif'%(simraw_file[:-4],str(megchan),cov_meth))
#            plt.close('all')
#            mlab.close(all=True)
#  %%%%%%%%%%%%%%
            #noise_cov.plot(epochs.info, show_svd=True, proj=True)
            #data_cov.plot(epochs.info, show_svd=True, proj=True)
            cov_rank = None if raw.info['proc_history']==[] else int(raw.info['proc_history'][0]['max_info']['sss_info']['nfree'])
            inverse_operator=mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, 
                                                                    rank=cov_rank, loose=1.0, depth=0.199)
            snr, snr_est = mne.minimum_norm.estimate_snr(evoked_pst, inverse_operator, verbose=True)
            if par['more_plots']=='sel11':
                plt.figure(dfname + '_snr&snr_est') 
                plt.plot(snr, 'r', label='snr'), plt.plot(snr_est, 'g', label='snr_est')
                plt.legend(loc='upper right', fontsize=None, shadow=False, framealpha=0.3)
                plt.suptitle(dfname + '_snr&snr_est')
                plt.savefig(out_path + dfname + '_snr&snr_est.png', facecolor='w', edgecolor='w', 
                            orientation='landscape', bbox_inches='tight', pad_inches=0.2)
            peak_ch, peak_time = evoked_pst.get_peak(ch_type='mag' if not megchan=='grad' else 'grad')
            tp = int((peak_time - par['actiwin'][0])*evoked_pst.info['sfreq'])
            snr_=snr[tp]
            print(snr_)
            # snr_est_mne = snr_est[tp]
            #mne.viz.plot_snr_estimate(evoked, inverse_operator, show=True)
            #plt.ylim(0,30)
            
            snr_mne, snr_est_mne, SNR_physio, SNR_golden, SNR_golden2, SNR_golden3 = estimate_3_snr(evoked_pst, evoked_pre, inverse_operator, fwd, src_amp=amp, megchan=megchan)
            snr_mne_10log10 = 10*np.log10(snr_mne) 
            if snr_mne<2:
                snr_mne2 = snr_mne - snr_mne/2
            elif snr_mne>2 and snr_mne<5:
                snr_mne2 = snr_mne - snr_mne/4
            elif snr_mne>5 and snr_mne<30:
                snr_mne2 = snr_mne
            elif snr_mne>30 and snr_mne<50:
                snr_mne2 = snr_mne + snr_mne/4
            elif snr_mne>50:
                snr_mne2 = snr_mne + snr_mne/2    
    
            snr_comp_file = '/net/qnap/data/rd/ChildBrain/BeamComp/Result_master/Simu_snr_comp_file_1.csv'
            fid = open(snr_comp_file, 'a+')
            fid.writelines('%s,'     %dfname)
            fid.writelines('%s,'     %str(amp))
            fid.writelines('%d,'     %label[0].vertices)
            fid.writelines('%.2f,'   %snr_)
            fid.writelines('%.2f,'   %snr_mne)
            fid.writelines('%.2f,'   %snr_est_mne)
            fid.writelines('%.2f,'   %SNR_physio)
            fid.writelines('%.2f,'   %SNR_golden)
            fid.writelines('%.2f,'   %SNR_golden2)
            fid.writelines('%.2f,'   %SNR_golden3)
            fid.writelines('%.2f,'   %snr_mne_10log10)
            fid.writelines('%.2f,'   %snr_mne2)
            fid.writelines('\n')
            fid.close()
            #mne.viz.plot_snr_estimate(evoked, inverse_operator, show=True)
            #plt.ylim(0,30)
            reg_form='SNR**4/500'
            #% %
            snr_def = 2.245 # to get the default reg=0.05 using reg_form
            ## SNR = snr_def

            for SNR in [snr_def]:#, snr_mne, SNR_physio, SNR_golden, SNR_golden2, snr_mne_10log10, snr_mne2]:
            #for reg in [reg_def, snr_mne**4/500, SNR_physio**4/500, SNR_golden**4/500, SNR_golden2**4/500]:    
                
                reg=eval(reg_form)
                print(SNR, reg)
                SNR=snr_mne
    
                #noise_cov.plot(epochs.info, show_svd=True, proj=True)
                #data_cov.plot(epochs.info, show_svd=True, proj=True)
                
    #                        stc = lcmv(evoked, fwd, noise_cov=noise_cov, data_cov=data_cov, reg=reg,
    #                                   pick_ori='max-power', max_ori_out='abs', reduce_rank=True, verbose=True)
    #                    kk, cnt =np.zeros((20,2)), -1
    #                    for reg in [0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
    #                        cnt = cnt + 1
#                rank=None#np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)-1 #np.linalg.matrix_rank(evoked.copy().crop(par['ctrlwin'][0], par['ctrlwin'][1]).data), 
#    #                        plt.figure('svd plot_' + dfname), plt.plot(np.linalg.svd(noise_cov.data)[1])
#                if 'filters' in locals():
#                    del filters
#                try:
#                    filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=reg,noise_cov=noise_cov,
#                                                          pick_ori='max-power', rank=rank,
#                                                          weight_norm='unit-noise-gain', reduce_rank=True, verbose=True)
#                except ValueError as Val_err:
#                    print('ValueError: Reduce rank manually > > > >\n\n')
#                    rank=np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)- 1
#                    # plt.figure('svd plot_' + dfname), plt.plot(np.linalg.svd(noise_cov.data)[1])
#                    filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=reg,noise_cov=noise_cov,
#                                                          pick_ori='max-power', rank=rank,
#                                                          weight_norm='unit-noise-gain', reduce_rank=True, verbose=True)
#                except np.linalg.LinAlgError as Lin_alg_err:
#                    print('LinAlgError: Reduce rank manually > > > >\n\n')
#                    rank=np.sum(np.linalg.svd(noise_cov.data)[1]>1e-30)- 1
#                    # plt.figure('svd plot_' + dfname), plt.plot(np.linalg.svd(noise_cov.data)[1])
#                    filters = mne.beamformer.make_lcmv(evoked.info, fwd, data_cov, reg=reg,noise_cov=noise_cov,
#                                                          pick_ori='max-power', rank=rank,
#                                                          weight_norm='unit-noise-gain', reduce_rank=True, verbose=True)
                rank = cov_rank #None
                #weight_norm = 'nai' # 'unit-noise-gain'
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
                    
    #        for reg in (0.05, eval(reg_form)):
    #            # reg = eval(reg_form)
    #            # reg = 0.05
    #            # stc = lcmv(evoked, fwd, noise_cov=noise_cov, data_cov=data_cov, reg=reg,
    #            #            pick_ori='max-power', max_ori_out='abs', reduce_rank=True, verbose=True)
    #            # kk, cnt =np.zeros((20,2)), -1
    #            # for reg in [0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
    #            #    cnt = cnt + 1
    #            rank = None# np.sum(np.linalg.svd(noise_cov.data)[1]>1e-27)-2 #np.linalg.matrix_rank(evoked.copy().crop(par['ctrlwin'][0], par['ctrlwin'][1]).data), 
    #            #                        plt.figure('svd plot_' + dfname), plt.plot(np.linalg.svd(noise_cov.data)[1])
    #            filters = mne.beamformer.make_lcmv(evoked_pst.info, fwd, data_cov, reg=reg, noise_cov=noise_cov,
    #                                                              pick_ori='max-power', rank=rank,
    #                                                              weight_norm='unit-noise-gain', reduce_rank=True, verbose=True)
                stc = mne.beamformer.apply_lcmv(evoked_pst, filters, max_ori_out='signed', verbose=True)
                #filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=reg, noise_cov=noise_cov)
                #stc = mne.beamformer.apply_lcmv(evoked_pst, filters)
                
                stc = np.abs(stc)
                # added to make robust >>>>>>>>>>
                n_reject_stc = len(stc.vertices)*1/100
                for nn in range(n_reject_stc):
                    src_peak, t_peak = stc.get_peak()
                    stc.data[np.where(stc.vertices==src_peak)]
                    
                
                
                src_peak, t_peak = stc.get_peak()
                timepoint = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)
                est_loc = fwd['src'][0]['rr'][src_peak]*1000 
                loc_err = np.sqrt(np.sum(np.square(stc_loc-est_loc)))  # Calculate loc_err
                print('Act_Sourceloc for %s' %dfname + '= %s' % str(stc_loc)) 
                print('Est_SourceLoc for %s' %dfname + '= %s' % str(np.around(est_loc,1)))
                print('Peak_Value for %s' %dfname + '= %.2f' % stc.data.max())
                print('Loc_error for %s' %dfname + '= %.1f' % loc_err)
                
                est_loc_mm = est_loc/1000
                # closest_index_, closest_point_ = closest_vert(est_loc_mm, surf_h['rr'])
                dist_est2scansurf = np.sqrt(np.sum(np.square(est_loc_mm-closest_point1)))*1000
                
                # closest_index_, closest_point_ = closest_vert(est_loc_mm, surf_simu_bem['rr'])
                dist_est2simusurf = np.sqrt(np.sum(np.square(est_loc_mm-closest_point2)))*1000
                                        
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
                loc_err2 =  np.sqrt(np.sum(np.square(stc_loc-(cntrd_locs1*1000))))
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
                    #vrtx = range(0,len(model[0]['rr']),2)
                    #innner_skull_head = np.dot(model[0]['rr'], trans_val[0]['trans'][:3, :3].T)
                    #innner_skull_head2 = mne.transforms.apply_trans(trans_val[0]['trans'].T, model[0]['rr'], move=True)
                    #mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 700))
                    #mlab.points3d(locs1[:,0], locs1[:,1], locs1[:,2],mode='sphere', scale_factor=0.005, color=(1,1,0))
                    #mlab.points3d(cntrd_locs1[:,0], cntrd_locs1[:,1], cntrd_locs1[:,2], mode='sphere', scale_factor=0.005, color=(0.9,0.8,0.9))
                    #mlab.points3d(innner_skull_head[vrtx,0], innner_skull_head[vrtx,1], innner_skull_head[vrtx,2], mode='sphere', scale_factor=.0005, color=(0.8,0.9,0.8))
                    #mlab.points3d(innner_skull_head2[vrtx,0], innner_skull_head2[vrtx,1], innner_skull_head2[vrtx,2], mode='sphere', scale_factor=.0005, color=(0.1,0.9,0.8))
                    ###mlab.show()
                    stc_loc_mm, est_loc_mm = stc_loc/1000, est_loc/1000
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
                    a = anim() # Starts the animation.
                    ###mlab.show()
    
                #% % Print results in result_file >>>>>>>>>>>>>>>>>>>>>>
                fid = open(resultfile, 'a+')
                if amp==amps[0] and vertnum==0 and round(reg,4)==0.0508:
                    fid.writelines('\n%s\n%s\n' %('**************', '****************'))
                fid.writelines('%s,'   %dfname)
                fid.writelines('%d,'   %amp)
                fid.writelines('%.2f,' %est_loc[0])
                fid.writelines('%.2f,' %est_loc[1])
                fid.writelines('%.2f,' %est_loc[2])
                fid.writelines('%.2f,' %stc.data.max())
                fid.writelines('%.2f,' %loc_err)
                fid.writelines('%.2f,' %np.sqrt(np.sum(np.square([0,0,0]-stc_loc))))
                fid.writelines('%.2f,' %np.sqrt(np.sum(np.square([0,0,0]-est_loc))))
                fid.writelines('%.f,'  %evoked_pst.nave)
                fid.writelines('%.f,'  %len(evoked.ch_names))
                fid.writelines('%.3f,' %SNR)
                fid.writelines('%.3f,' %reg)
                fid.writelines('%s,'   %str(rank))
                fid.writelines('%s,'   %'')
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
                fid.writelines('%d,'   %label[0].vertices)
                fid.writelines('%.2f,' %dist_act2scansurf) # distance fron actual source to scanning bem surf
                fid.writelines('%.2f,' %dist_est2scansurf) # distance fron estimated source to scanning bem surf
                fid.writelines('%.2f,' %dist_act2simusurf) # distance fron actual source to simulation bem surf
                fid.writelines('%.2f,' %dist_est2simusurf) # distance fron estimated source to simulation bem surf
                fid.writelines('%s,'   %cov_meth)
                fid.writelines('%d,'   %vertnum)
                fid.writelines('%d,'   %(t_peak*1000))
                fid.writelines('%s,\n' %weight_norm)
                fid.close()

                if par['result_plot']=='yess':
                    plt.figure()
                    ts_show = -163  # show first 5 peak sources 
                    plt.plot(1e3 * stc.times, stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
                    plt.title(dfname + ' for %d largest sources'%abs(ts_show))
                    plt.xlabel('time (ms)')
                    plt.ylabel('%s value ' %'LCMV stc'+ '@reg=%.2f'%reg )
                    plt.show()
                    if par['save_resplot']=='yess':
                        plt.savefig(out_path + dfname + '_STCplot.png', facecolor='w', edgecolor='w', 
                                    orientation='landscape', bbox_inches='tight', pad_inches=0.2)
                
                if par['result_plot']=='yessss' and not par['SL_cort']=='yes': # Volumetric plot
                    thresh = stc.data.max()*0.50
                    img=mne.save_stc_as_volume('lcmv_inverse.nii.gz', stc, fwd['src'], dest='mri', mri_resolution=False)
                    plot_stat_map(index_img(img, timepoint), mrifile, threshold=thresh)
                    plt.suptitle('%s'%dfname + ' /LCMV (tpeak=%.3f s.)\n' % stc.times[timepoint] + 
                                 'PeakValue= %.3f' % stc.data.max() + ' / Reg= %.3f' % reg + 'Est_loc= %.1f' % est_loc[0] + 
                                 ', %.1f' % est_loc[1]+', %.1f ' % est_loc[2] + '/ Loc_err= %.2f mm' % loc_err, 
                                 fontsize=12, color='white')
                    #plt.pause(1.0), manager=plt.get_current_fig_manager(), manager.window.showMaximized()
                    if par['save_resplot']=='yes':
                        plt.savefig(out_path + dfname + '_Sourceplot.png', dpi=100, facecolor='w', edgecolor='w',
                                    orientation='landscape', papertype=None, format=None, transparent=False,
                                     bbox_inches='tight', pad_inches=0.2, frameon=None)
                    remove('lcmv_inverse.nii.gz')
                elif par['result_plot']=='yes' and par['SL_cort']=='yes': # Cortically constrainned plot
                    brain=stc.plot(subject=subject, surface='inflated', hemi='both', colormap='auto', time_label='auto', 
                         smoothing_steps=10, transparent=None, alpha=1.0, time_viewer=False, subjects_dir=subjects_dir, 
                         figure=None, views='lat', colorbar=True, clim='auto', cortex='classic', size=800, 
                         background='black', foreground='white', initial_time=t_peak, time_unit='s', 
                         backend='auto', spacing='oct5')
                    if stc_loc[0]<=0:
                        brain.add_foci(act_src_vert, coords_as_verts=True, hemi='lh', color='blue')
                    else:
                        brain.add_foci(act_src_vert, coords_as_verts=True, hemi='rh', color='blue')
                    if est_loc[0]<=0:
                        brain.add_foci(src_peak, coords_as_verts=True, hemi='lh', color='red')
                    else:
                        brain.add_foci(src_peak, coords_as_verts=True, hemi='rh', color='red')
                    mlab.view(0, 0, 550, [0, 0, 0])
                    mlab.title('LCMV.......', height=0.9)
                
                plt.close('all')
                mlab.close(all=True)

# ***************************************************************************** #!/usr/bin/env python2
