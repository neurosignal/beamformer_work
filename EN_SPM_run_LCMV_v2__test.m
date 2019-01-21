% function EN_SPM_run_LCMV_v2__test(varargin)
function EN_SPM_run_LCMV_v2__test(datacov,out_tmp_dir, D, gridres, bpfreq,...
                                  reg, woi, chan_categories, meg, cov_meth)
%% for data    
    clear matlabbatch; tic 
    matlabbatch{1}.spm.tools.beamforming.data.dir = {pwd};
    matlabbatch{1}.spm.tools.beamforming.data.D = {fullfile(D)};
    matlabbatch{1}.spm.tools.beamforming.data.val = 1;
    matlabbatch{1}.spm.tools.beamforming.data.gradsource = 'inv';
    matlabbatch{1}.spm.tools.beamforming.data.space = 'Head'; 
    matlabbatch{1}.spm.tools.beamforming.data.overwrite = 1;
    spm_jobman('run', matlabbatch); toc

%% for source    
    clear matlabbatch; tic 
    matlabbatch{1}.spm.tools.beamforming.sources.BF = {[out_tmp_dir '/BF.mat']};
    %%matlabbatch{2}.spm.tools.beamforming.sources.BF(1) = cfg_dep('Prepare data: BF.mat file', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{1}.spm.tools.beamforming.sources.reduce_rank = [2 3];
    matlabbatch{1}.spm.tools.beamforming.sources.keep3d = 1;
    % matlabbatch{1}.spm.tools.beamforming.sources.plugin.grid_phantom.resolution = par.gridres;
    matlabbatch{1}.spm.tools.beamforming.sources.plugin.grid.resolution = gridres;
    matlabbatch{1}.spm.tools.beamforming.sources.plugin.grid.space = 'Head';
    matlabbatch{1}.spm.tools.beamforming.sources.visualise = 1;
    spm_jobman('run', matlabbatch); toc

%% for covariance matrix
    clear matlabbatch; tic 
    matlabbatch{1}.spm.tools.beamforming.features.BF = {[out_tmp_dir '/BF.mat']};
    %%matlabbatch{3}.spm.tools.beamforming.features.BF(1) = cfg_dep('Define sources: BF.mat file', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{1}.spm.tools.beamforming.features.whatconditions.all = 1;
    matlabbatch{1}.spm.tools.beamforming.features.woi = woi;
    if strcmp(chan_categories{1,meg}, 'all')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {
                                                                  'MEG'
                                                                  'MEGPLANAR'
                                                                  }';
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'meg';          % all
    elseif strcmp(chan_categories{1,meg}, 'mag')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {'MEGMAG'}; %'MEG' check this if not works
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'no';
    elseif strcmp(chan_categories{1,meg}, 'grad')
        matlabbatch{1}.spm.tools.beamforming.features.modality = {'MEGPLANAR'};
        matlabbatch{1}.spm.tools.beamforming.features.fuse = 'no';
    end
    
    if isequal(cov_meth, 'cov')             %% Sample cov ......................
        matlabbatch{1}.spm.tools.beamforming.features.plugin.cov.foi = bpfreq;
        matlabbatch{1}.spm.tools.beamforming.features.plugin.cov.taper = 'hanning';
    elseif isequal(cov_meth, 'contcov')     %% Robust covariance ...............
        matlabbatch{8}.spm.tools.beamforming.features.plugin.contcov = struct([]);
    elseif isequal(cov_meth, 'csd')         %% CSD method for dics..............
        matlabbatch{1}.spm.tools.beamforming.features.plugin.csd.foi = '<UNDEFINED>';
        matlabbatch{1}.spm.tools.beamforming.features.plugin.csd.taper = 'dpss';
        matlabbatch{1}.spm.tools.beamforming.features.plugin.csd.keepreal = 0;
        matlabbatch{1}.spm.tools.beamforming.features.plugin.csd.hanning = 1;
    elseif isequal(cov_meth, 'regmulticov') %% Regularized multiple covariance....
        matlabbatch{1}.spm.tools.beamforming.features.plugin.regmulticov = 1;
    end
    matlabbatch{1}.spm.tools.beamforming.features.regularisation.manual.lambda = reg;
    matlabbatch{1}.spm.tools.beamforming.features.bootstrap = false;
    spm_jobman('run', matlabbatch); toc
    
    if ~isempty(datacov) % for using externally defined cov matrix
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

%% For Inverse solution
    clear matlabbatch; tic
    matlabbatch{1}.spm.tools.beamforming.inverse.BF = {[out_tmp_dir '/BF.mat']};
%     matlabbatch{4}.spm.tools.beamforming.inverse.BF(1) = cfg_dep('Covariance features: BF.mat file', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));       
    matlabbatch{1}.spm.tools.beamforming.inverse.plugin.lcmv.orient = true;
    matlabbatch{1}.spm.tools.beamforming.inverse.plugin.lcmv.keeplf = false;
    spm_jobman('run', matlabbatch); toc
    
%% For output 
    clear matlabbatch; tic
    matlabbatch{1}.spm.tools.beamforming.output.BF = {[out_tmp_dir '/BF.mat']};
    %%matlabbatch{5}.spm.tools.beamforming.output.BF(1) = cfg_dep('Inverse solution: BF.mat file', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.powermethod = 'trace';
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.whatconditions.all = 1;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.sametrials = false;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.woi = woi;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.foi = bpfreq;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.contrast = [-1 1];
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.logpower = true;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.result = 'singleimage'; %  bycondition
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.scale = 2;
    matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG';
    if strcmp(chan_categories{1,meg}, 'all')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG';
    elseif strcmp(chan_categories{1,meg}, 'mag')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEG'; 
    elseif strcmp(chan_categories{1,meg}, 'grad')
        matlabbatch{1}.spm.tools.beamforming.output.plugin.image_power.modality = 'MEGPLANAR';
    end
    %     matlabbatch{6}.spm.tools.beamforming.write.BF(1) = cfg_dep('Output: BF.mat file', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','BF'));
    %     matlabbatch{6}.spm.tools.beamforming.write.plugin.nifti.normalise = 'separate';
    %     matlabbatch{6}.spm.tools.beamforming.write.plugin.nifti.space = 'native';%'mni'; 
    %     matlabbatch{7}.spm.util.disp.data(1) = cfg_dep('Write: Output files', substruct('.','val', '{}',{6}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
    spm_jobman('run', matlabbatch); toc
                
     % saveas(gcf,['SPM_' mfname '_sss' num2str(sss) '-' num2str(par.runtry) '-LCMV'],'png');
     % overlay=ft_read_mri(['uv_pow_effspmeeg_' mfname '.nii'])           
end