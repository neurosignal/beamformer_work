function EN_SPM_run_LCMV2_v2__test(datacov, out_tmp_dir, bpfreq, reg, woi, chan_categories, megg)
%   matlabbatch{1,3}=[];
%   matlabbatch{1,4}=[];
%   matlabbatch{1,5}=[];
%   matlabbatch{1,6}=[];
%   matlabbatch{1,7}=[];
    clear matlabbatch
    matlabbatch{1}.spm.tools.beamforming.features.BF = {[out_tmp_dir '/BF.mat']};
    matlabbatch{1}.spm.tools.beamforming.features.whatconditions.all = 1;
    matlabbatch{1}.spm.tools.beamforming.features.woi = woi;
    if strcmp(chan_categories{megg,1}, 'all')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {
                                                                  'MEG'
                                                                  'MEGPLANAR'
                                                                  }';
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'meg'; %all
    elseif strcmp(chan_categories{megg,1}, 'mag')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {'MEGMAG'}; %'MEG' check this if not works
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'no';
    elseif strcmp(chan_categories{megg,1}, 'grad')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {'MEGPLANAR'};
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'no';
    end
    matlabbatch{1}.spm.tools.beamforming.features.plugin.cov.foi = bpfreq;
    matlabbatch{1}.spm.tools.beamforming.features.plugin.cov.taper = 'hanning';
    matlabbatch{1}.spm.tools.beamforming.features.regularisation.manual.lambda = reg; %par.reg;
    matlabbatch{1}.spm.tools.beamforming.features.bootstrap = false;
    spm_jobman('run', matlabbatch);
    
    if ~isempty(datacov)
        fprintf('\nNote: Injecting covariance matrix defined externally (from MNE-Python: %s). . . . . \n',datacov.cov_meth)
        BF = load([out_tmp_dir '/BF.mat']);
        features = BF.features;
        data     = BF.data;
        sources  = BF.sources;
        clear BF
        features.MEG = rmfield(features.MEG, 'C');
        features.MEG = rmfield(features.MEG, 'Cinv');
        C = datacov.data;
        lambda = (reg/100) * trace(C)/size(C,1);
        C      = C + lambda * eye(size(C));
        Cinv   = pinv_plus(C);
        % U      = eye(size(C));
        features.MEG.C    = C;
        features.MEG.Cinv = Cinv;
        % features.U    = U;
        delete([out_tmp_dir '/BF.mat'])
        save([out_tmp_dir '/BF.mat'], 'data','features', 'sources')
    end
    
    clear matlabbatch; tic
    matlabbatch{1}.spm.tools.beamforming.inverse.BF = {[out_tmp_dir '/BF.mat']};
    %%matlabbatch{4}.spm.tools.beamforming.inverse.BF(1) = cfg_dep('Covariance features: BF.mat file', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));       
    matlabbatch{1}.spm.tools.beamforming.inverse.plugin.lcmv.orient = true;
    matlabbatch{1}.spm.tools.beamforming.inverse.plugin.lcmv.keeplf = false;
    spm_jobman('run', matlabbatch); toc
    
    clear matlabbatch; tic
    matlabbatch{1}.spm.tools.beamforming.output.BF = {[out_tmp_dir '/BF.mat']};
    %%matlabbatch{3}.spm.tools.beamforming.output.BF(1) = cfg_dep('Inverse solution: BF.mat file', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.powermethod = 'trace';
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.whatconditions.all = 1;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.sametrials = false;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.woi = woi;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.foi = bpfreq;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.contrast = [-1 1];
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.logpower = true;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.result = 'singleimage'; %  bycondition
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.scale = 2;
    if strcmp(chan_categories{1,megg}, 'all')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG';
    elseif strcmp(chan_categories{1,megg}, 'mag')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG'; 
    elseif strcmp(chan_categories{1,megg}, 'grad')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEGPLANAR';
    end
        %%matlabbatch{4}.spm.tools.beamforming.write.BF(1) = cfg_dep('Output: BF.mat file', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    % % matlabbatch{4}.spm.tools.beamforming.write.BF(1) = cfg_dep('Output: BF.mat file', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    % % matlabbatch{4}.spm.tools.beamforming.write.plugin.nifti.normalise = 'separate';
    % % matlabbatch{4}.spm.tools.beamforming.write.plugin.nifti.space = 'native';%'mni';
    % % 
    % % matlabbatch{5}.spm.util.disp.data(1) = cfg_dep('Write: Output files', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));

    spm_jobman('run', matlabbatch); toc
    
    % saveas(gcf,['SPM_' mfname '_sss' num2str(sss) '-' num2str(par.runtry) '-LCMV'],'png');
    % overlay=ft_read_mri(['uv_pow_effspmeeg_' mfname '.nii'])
            
end