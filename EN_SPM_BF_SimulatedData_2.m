  %% FieldTrip simulated data beamforming pipeline (Combining LCMV and DICS)
%% Date: 26/07/2018
%% Author: Amit @ Elekta Neuromag
%% Add fieldtrip and spm in path
clearex leadfield headmodel; % for FT leadfield & headmodel comparison
clc; refresh
restoredefaultpath
add_ft='yes'; add_spm='yes';
if ispc, homedir = 'C:\Users\fijaiami//'; 
else  homedir = '/home/amit//'; end
addpath([homedir 'git//ChildBrain//BeamComp//MATLAB//'])
EN_add_toolboxes(add_ft, add_spm);
cd([homedir 'spm_temp//'])
close; tic;

%% Set data directory and other parameters
pick_datalines = 24385:24785;%:27000;%24523%26740:27000;%15051:20252;%15051:17451;
if ispc
    data_path = '\\172.16.50.206\data\rd\ChildBrain\Simulation\';
    out_path  = 'C:\Users\fijaiami\git//ChildBrain//BeamComp\BeamComp_Resultfiles\Simulations//';
    mri_path  = 'C:\Users\fijaiami\Documents\Visits&Purchases\BirmUni-Feb2018\Training\multimodal_data\MRI\';
elseif isunix
    data_path = '/net/qnap/data/rd/ChildBrain/Simulation/NEW_25_seglabBEM6/';%NEW_20/';
    out_path  = '/home/amit/git//ChildBrain//BeamComp/BeamComp_Resultfiles/Simulations/';
    out_path1 = '/net/qnap/data/rd/ChildBrain/TEMP4SPM/';
    mri_path  = '/net/qnap/data/rd/ChildBrain/neurodata/beamformer_data/Beamformer_share/multimodal/sub1/MRI/';
end
par                 = [];
par.cov_meth        = 'spm_sample_cov';%'shrinkage'%;%
par.prep            = {'', '_sss', '_tsss', '_tsss_mc','_cxsss', '_nosss'};
par.meg             = {'all', 'mag', 'grad'};
par.visual          = 'no';
par.powspect        = 'no';
par.browse          = 'no'; 
par.more_plots      = 'yes';
par.source_ortho    = 'yes__';
par.resultplotsave  = '';
par.plot_stc        = 'yes__';
par.runtry          = 1;
par.gridres         = 5.0; % in mm
par.bpfreq          = [2 40];
par.bsfreq          = [49.1 50.9];
par.cov_cut         = [0.01 98];
par.zscore_cut      = [0.01 100];
par.stimchan        = 'STI101';
par.mri_seg         = 'no';
par.apply_pca       = 'no';
par.reg_form        = 'SNR^4/500';
par.apply_lcmv      = 'yes';
par.apply_dics      = 'no';
par.reg_compare     = '';
data_to_filter      = 'continuous'; %  % epoched % % continuous is highly recomended
transit             = 'par.trialwin(2)*0.1'; % 10%

%% Read row-wise entries from first column from the result csv file
inputfile  = [out_path 'Simulations_Est_SourceLoc_1.csv'];
result_figfile  = [out_path 'SPM_result_figfile.doc'];
mrifname    = [mri_path 'nenonen_jukka.nii,1'];
fnamepre    = ''; %just to initialize
%%
for linenum = pick_datalines
    for meg = {'all'}%,'mag', 'grad'}
        for sss = 0
            if isunix
%                 endline = 1;
%     %             while linenum>endline
%     %                 fid         = fopen(inputfile);spydergfdhtr
%     %                 track_data  = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s ','delimiter', ',', 'EmptyValue', -Inf);
%     %                 fclose(fid);
%     %                 disp('wait ...................')
%     %                 endline = length(track_data{1,1});
%     %                 pause(1)
%     %             end
%     %             disp('proceed >>>>>>>>>>>>>>>>>>')
%     %             dfname      = track_data{1,1}{linenum,1};
%     %             fname       = [data_path dfname '.fif'];
%     %             act_loc     = regexp(dfname,'(?<=_at_).*(?=mm_OVER_)','match');
%     %             act_loc     = regexp(act_loc, '_', 'split');
%     %             act_loc     = cellfun(@str2double,act_loc{1,1});
%     %             SNR         = str2double(track_data{1,12}{linenum,1});
%     %             amp         = str2double(track_data{1,2}{linenum,1});
% 
%                 while linenum>endline
%                     clear track_data
%                     track_data  = importdata(inputfile);
%                     endline = length(track_data);
%                     disp('wait ...................')
%                 end
%                 disp('proceed >>>>>>>>>>>>>>>>>>')
%                 track_data = regexp(track_data{linenum,1}, ',', 'split');
%                 dfname      = [track_data{1,1}];%(1:end-5) '_sss'];
%                 fname       = [data_path dfname '.fif'];
%                 act_loc     = regexp(dfname,'(?<=_at_).*(?=mm_OVER_)','match');
%                 act_loc     = regexp(act_loc, '_', 'split');
%                 act_loc     = cellfun(@str2double,act_loc{1,1});
%                 SNR         = str2double(track_data{1,12});
%                 amp         = str2double(track_data{1,2});
            
            
                endline = 1;
                while linenum>endline
                    fid         = fopen(inputfile);
                    track_data  = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s ','delimiter', ',', 'EmptyValue', -Inf);
                    fclose(fid);
                    disp('wait ...................')
                    endline = length(track_data{1,1});
                    pause(1)
                end
% %                 for mm=1:size(track_data,2) % just to avoid alternate extra line
% %                     track_data{1,mm}= track_data{1,mm}(1:2:length(track_data{1,end}));
% %                 end
                disp('proceed >>>>>>>>>>>>>>>>>>')

                dfname      = track_data{1,1}{linenum,1};
                fname       = [data_path dfname '.fif'];
                act_loc     = regexp(dfname,'(?<=nAm_at_).*(?=mm)','match');
                act_loc     = regexp(act_loc, '_', 'split');
                act_loc     = cellfun(@str2double,act_loc{1,1});
                SNR         = str2double(track_data{1,12}{linenum,1});
                amp         = str2double(track_data{1,2}{linenum,1});
                loc         = act_loc;
                vertnum     = str2double(track_data{1,31}{linenum,1});
                LocErr_     = str2double(track_data{1,7}{linenum,1});

                snr_def     =  SNR;
                snr_mne     = SNR%str2double(track_data{1,12}{linenum+1,1});
%                 SNR_physio  = str2double(track_data{1,12}{linenum+2,1});
%                 SNR_golden  = str2double(track_data{1,12}{linenum+3,1});
%                 SNR_golden2 = str2double(track_data{1,12}{linenum+4,1});
%                 snr_mne_10log10 = 10*log10(snr_mne);

                clear track_data
            end
        
            fprintf('Filename = %s \nAmplitude = %dnAm \nSNR = %.3f\n', dfname, amp, snr_mne)
            disp(act_loc)
            clear track_data

            load([homedir 'git//ChildBrain//BeamComp/MATLAB/channelslist.mat']);
            if isequal(meg, {'all'})
                ignorech = [];
            elseif isequal(meg, {'mag'})
                ignorech = channelslist.grad;
            elseif isequal(meg, {'grad'})
                ignorech = channelslist.mag;
            end
            if isequal(fname(end-6:end-4), 'raw') || isequal(fname(end-8:end-4), 'nossp')
                % bads     = {'MEG2233'}; 
                bads = {'MEG2233', 'MEG2212','MEG2332', 'MEG0111'};
            else
                bads     = {};
            end
            badch        = [ignorech, bads];
            par.trialwin = [-0.200 0.200];
            par.ctrlwin  = [-0.100 -0.010];
            par.actiwin  = [0.010 0.100];
            par.win_oi   = [-0.100, -0.010; 0.010, 0.100];
            woi          = par.win_oi*1000;     % convert into milisecond
            trial_win    = par.trialwin*1000;  % convert into milisecond
            
            par.badtrs   = [15,103,104,107]; % added on 02/12/18
            
            megchan=channelslist.meg;
            megchan(ismember(megchan, badch))=[]; 

            % Find trigger categories && label them
            keyset = {'Simulated'};  valueset = 5;
            evdict = containers.Map(keyset, valueset);

            if ~isequal(fnamepre, fname) % && LocErr_>10
                disp(linenum)
                fnamepre = fname;
                disp(dfname) 
                disp(act_loc)

                %% Browse raw data
                if isequal(par.browse,'yes')
                    cfg          = [];
                    cfg.channel  = [megchan par.stimchan];
                    cfg.viewmode = 'butterfly';
                    cfg.blocksize= 15;
                    cfg.ylim     = [-1e-11 1e-11];%'maxmin';
                    cfg.demean   = 'yes';
                    cfg.dataset  = fname;
                    ft_databrowser(cfg);
                end
                
                %% Define trial definition matrix
                trls = EN_make_SPM_trialdef_realdata(fname, par.stimchan,...
                                        valueset, par.trialwin, 0, 0);
                if isequal(par.browse, 'yes')
                    cfg        = [];
                    trlss      = trls;
                    trlss(:,4) = valueset;
                    cfg.trl    = trlss;
                    ft_plot_events(figure, cfg, keyset, valueset)
                end  
                %% Convert fif raw data to SPM epoched format 
                if isequal(data_to_filter, 'epoched')
                    S                 = [];
                    S.dataset         = fname;
                    S.mode            = 'epoched';
                    S.channels        = megchan; %[megchan, {par.stimchan}];
                    S.saveorigheader  = 1;
                    S.inputformat     = 'neuromag_fif'; %%%%%%%%%%%%
                    S.conditionlabels = char(keyset);
                    S.outfile         = ['spmeeg_' dfname];
                    S.trl             = trls;
                    D = spm_eeg_convert(S);
                end
                
                %% To read in continuous mode        
                if isequal(data_to_filter, 'continuous')
                    S                   = [];
                    S.dataset           = fname;
                    S.mode              = 'continuous';
                    S.channels          = megchan; %[megchan, {par.stimchan}];
                    S.saveorigheader    = 1;
                    S.inputformat       ='neuromag_fif'; %%%%%%%%%%%%
                    S.conditionlabels   = char(keyset);
                    S.outfile           = ['spmeeg_' dfname];
                    D = spm_eeg_convert(S);
                end    
%% Baseline correction and DC removal
                    S   = [];
                    S.D = D;
                    % S.timewin - 2-element vector with start and end of baseline period [ms]
                    S.prefix ='bc';
                    D = spm_eeg_bc(S);
                %% Reject bad channels 
                % D = badchannels(D, D.indchannel(badch), 1); %(don't do again if removed from channel list)
                % save(D);
                %% Filter the converted raw data (band pass and bandstop)
                S       = [];
                S.D     = D;
                S.type  = 'butterworth';
                S.band  = 'bandpass';
                S.freq  = par.bpfreq;
                S.dir   = 'twopass';
                S.prefix= 'f';
                D = spm_eeg_filter(S);
                
                if par.bpfreq(2)>48
                    S       = [];
                    S.D     = D;
                    S.type  = 'butterworth';
                    S.band  = 'stop';
                    S.freq  = [49 51];
                    S.dir   = 'twopass';
                    S.prefix= 'f';
                    D = spm_eeg_filter(S); 
                end
                % Save data to use in MNE, if needed
                spm_data = D.ftraw;
                spm_data.hdr.orig = D.origheader;
                fieldtrip2fiff(sprintf('%sSPMread_%s_bp_%d-%dHz.fif',data_path, dfname, par.bpfreq), spm_data); clear spm_data
                
                %% Epoch the filtered continuous data >>>>>>>>
                if isequal(data_to_filter, 'continuous') % if the filtered data was continuous
                    S                 = [];
                    S.D               = D;           
                    S.bc              = 1;
                    S.trl             = trls(:,1:3); % [N x 3] trl matrix or name of the trial definition 
                    S.conditionlabels = keyset;      % labels for the trials in the data 
                    S.prefix          = 'e';         % prefix for the output file (default - 'e')
                    D = spm_eeg_epochs(S);
                end
        
                %% Find trial variance && z-score outliers and index them
                [selecttrials, par] = NM_ft_varcut2(D.ftraw, par);
                %%
% % %                 clear trl_var trl_zscore
% % %                 trlindx = 1:D.ntrials;
% % %                 timelk = D.fttimelock.trial;
% % %                  for trl = 1:D.ntrials
% % %                      trl_var(trl,:)= max(var(squeeze(timelk(trl,:,:))'));
% % %                      trl_zscore(trl,:)= max(max(zscore(squeeze(timelk(trl,:,:))')));
% % %                  end
% % %                 percentiles = prctile(trl_var, par.cov_cut);
% % %                 outlr_idx = trl_var < percentiles(1) | trl_var > percentiles(2);
% % %                 bd_trl_var = trlindx(outlr_idx);
% % %                 percentiles = prctile(trl_zscore, par.zscore_cut);
% % %                 outlr_idx = trl_zscore < percentiles(1) | trl_zscore > percentiles(2);
% % %                 bd_trl_zscore = trlindx(outlr_idx);
% % %                 bd_trls = union(bd_trl_var, bd_trl_zscore);
% % %                 par.bad_trials = [par.badtrs, bd_trls];   par.badtrs = [];
% % %                 
% % %                 if isequal(par.more_plots, 'yes')           
% % %                     figure
% % %                     subplot(211),scatter(1:D.ntrials, trl_var, 50, 'b*'); xlim([1 D.ntrials]), title('Max. variance'), hold on
% % %                     subplot(212),scatter(1:D.ntrials, trl_zscore, 50, 'g*'); xlim([1 D.ntrials]), title('Max. z-score')
% % %                     subplot(211),scatter(par.bad_trials, trl_var(par.bad_trials), 70, 'ro', 'linewidth',2); 
% % %                     xlim([1 D.ntrials]), title('Max. variance')
% % %                 end
                %% Redefine trails to avoid the filter edge effect
                % S=[];
                % S.D = D;
                % S.timewin = actual_win;%[-400 400]; %%%%%%%%%%%%%%%%%%%%%%
                % S.prefix='choped_';
                % D = spm_eeg_crop(S);
                %% Browse converted raw data
                if isequal(par.browse, 'yes')
                    cfg             = [];
                    cfg.channel     = megchan;
                    cfg.viewmode    = 'vertical';
                    % cfg.blocksize   = abs(par.trialwin(1)) + par.trialwin(2);
                    %cfg.preproc.demean      = 'yes';
                    % ft_databrowser(cfg, D.ftraw);

                    cfg.viewmode    = 'butterfly';
                    ft_databrowser(cfg, D.ftraw);
                end
                %% Remove bad trials
                if par.bad_trials
                    D = badtrials(D, par.bad_trials, 1);
                    save(D);
                    S=[];
                    S.D=D;
                    S.prefix='r';
                    D=spm_eeg_remove_bad_trials(S);
                    fprintf('\nRemaining #trials = %d - %d = %d trials .........\nRemoved trials : ',...
                            size(trls,1), length(par.bad_trials), D.ntrials);   disp(par.bad_trials)
                end
                %%  Power Spectrum analysis and visualization (Additional)
                if isequal(par.powspect, 'yes')
                    cfg              = [];
                    cfg.output       = 'pow'; 
                    cfg.channel      =  megchan;
                    cfg.method       = 'mtmconvol';
                    cfg.taper        = 'hanning';
                    cfg.toi          = par.trialwin(1):0.01:par.trialwin(2);
                    cfg.foi          = par.bpfreq(1):2:par.bpfreq(2);
                    cfg.t_ftimwin    = ones(size(cfg.foi)) * 0.01;
                    cfg.trials       = find(D.ftraw.trialinfo(1,:) == 1);
                    TFR     = ft_freqanalysis(cfg, D.ftraw);

                    figure,
                    cfg = [];
                    cfg.baseline     = [par.trialwin(1) 0.0];
                    cfg.baselinetype = 'absolute'; 
                    cfg.showlabels   = 'no';  
                    cfg.layout       = 'neuromag306mag.lay';
                    subplot_tight(1,2,1,0.05), ft_multiplotTFR(cfg, TFR);
                    title([dfname '-Mags'], 'Interpreter', 'none', 'fontsize', 15)
                    cfg.layout       = 'neuromag306planar.lay';
                    subplot_tight(1,2,2,0.05), ft_multiplotTFR(cfg, TFR);
                    title([dfname '-Grads'], 'Interpreter', 'none', 'fontsize', 15)
                end   
                clear TFR 
                %% Evoked responce
                if isunix
                    addpath('/net/bonsai/home/amit/Documents/MATLAB/fieldtrip-18092018/src')
                end
                cfg = []; cfg.covariance='yes';
                cfg.covariancewindow = 'all';
                cfg.vartrllength = 2;
                cfg_ = []; cfg_.toilim = par.trialwin;
                evoked = ft_timelockanalysis(cfg, ft_redefinetrial(cfg_,D.ftraw));

                cfg = [];   cfg.covariance='yes';
                cfg.covariancewindow = 'all'; 
                cfg.vartrllength = 2;
                cfg_ = []; cfg_.toilim = par.ctrlwin;
                evokedpre = ft_timelockanalysis(cfg, ft_redefinetrial(cfg_,D.ftraw));

                cfg = [];   cfg.covariance='yes';
                cfg.covariancewindow = 'all'; 
                cfg.vartrllength = 2;
                cfg_ = []; cfg_.toilim = par.actiwin;
                evokedpst = ft_timelockanalysis(cfg, ft_redefinetrial(cfg_,D.ftraw));
                
                % SVD & rank
                [~,noise_svd,~] = svd(evokedpre.cov);
                [~,data_svd,~]  = svd(evokedpst.cov);
                noisecov_rank   = rank(evokedpre.cov); %np.linalg.matrix_rank(M=noise_cov['data'], tol=None)
                datacov_rank    = rank(evokedpst.cov); %np.linalg.matrix_rank(M=data_cov['data'], tol=None)
%             figure, subplot(211), plot(sum(noise_svd)), legend('Noise Cov.SVD'), xlim([0,306])
%                     subplot(212), plot(sum(data_svd)), legend('Data Cov.SVD'), xlim([0,306])
                %% Evoked responce (Optional)
                if isequal(par.more_plots, 'yes')           
                    figure('color', [1 1 1]), fs=15;
                    subplot_tight(4,4,[1,2],[0.06 0.015]);  
                    plot(evoked.time, evoked.avg); xlim([evoked.time(1) evoked.time(end)])
                    title(dfname, 'FontSize', fs, 'Interpreter', 'none')
                    subplot_tight(4,4,[5,6],[0.06 0.015]); clear trl_var; ftraw = D.ftraw;
                    for trl = 1:size(ftraw.trial,2), trl_var(trl,:) = max(var(ftraw.trial{1,trl}(:,:)'));  end
                    scatter(1:size(ftraw.trial,2), trl_var, 50, 'go', 'filled'); xlim([0 size(ftraw.trial,2)+1]), box on
                    title('Max. variance for all selected trials', 'FontSize', fs), clear ftraw          
                    subplot_tight(2,4,3,[0.06 0.001]);
                    imagesc(evokedpre.cov), title(['Noise Cov [' num2str(min(min(evokedpre.cov))) ' ' num2str(max(max(evokedpre.cov))) ']'], 'FontSize', fs)
                    subplot_tight(2,4,4,[0.06 0.001]);
                    imagesc(evokedpst.cov), title(['Data Cov [' num2str(min(min(evokedpst.cov))) ' ' num2str(max(max(evokedpst.cov))) ']'], 'FontSize', fs)
                    subplot_tight(2,4,5,[0.06 0.001]);   
                    cfg = []; cfg.layout = 'neuromag306mag.lay';
                    ft_multiplotER(cfg, evoked); 
                    title('Magnetometer', 'FontSize', fs)
                    subplot_tight(2,4,6,[0.06 0.001]);   
                    cfg = []; cfg.layout = 'neuromag306planar.lay';
                    ft_multiplotER(cfg, evoked); 
                    title('Gradiometer', 'FontSize', fs)
                    cfg=[];  cfg.method = 'mtmfft'; cfg.output = 'pow';
                    cfg.foi  = par.bpfreq(1):2:par.bpfreq(2); cfg.taper = 'hanning';
                    tfr= ft_freqanalysis(cfg, evokedpst);
                    subplot_tight(2,4,7,[0.06 0.001]);
                    cfg = []; cfg.layout = 'neuromag306mag.lay'; 
                    ft_topoplotTFR(cfg, tfr); title('Post-stim Mags', 'FontSize', fs)
                    subplot_tight(2,4,8,[0.06 0.001]);
                    cfg = []; cfg.layout = 'neuromag306planar.lay';
                    ft_topoplotTFR(cfg, tfr); title('Post-stim Grads', 'FontSize', fs), clear tfr
                    set(gcf, 'Position', get(0, 'Screensize'));  
                    
                    img = getframe(gcf);
                    imwrite(img.cdata, [data_path 'SPM//' dfname '-data.jpeg']), clear img; %close()
%                     mkdir([data_path 'SPM'])
%                     saveas(gcf, [data_path 'SPM//' dfname '-data.jpeg'], 'jpg'), close
                end

                %% Apply TSSS on raw data
                if sss
                    S = [];
                    S.D = D;
                    S.tsss       = 1;     
                    S.t_window   = 1;     
                    S.corr_limit = 0.98;  
                    S.magscale   = 59.5;     
                    S.xspace     = 0;     
                    S.Lin        = 8;     
                    S.Lout       = 3;     
                    S.cond_threshold = 50; 
                    S.prefix     = 'sss_'; 
                    D = tsss_spm_enm(S);

                    D = badchannels(D, ':', 0);  save(D);

                    if isequal(par.more_plots, 'yes')
                        cfg             = [];
                        cfg.viewmode    = 'butterfly';
                        cfg.blocksize   = 15;
                        ft_databrowser(cfg, D.ftraw);     
                    end 
                else
                    D = montage(D, 'switch', 0);save(D);
                end
                % Run the batch    
                clear matlabbatch
                if sss
                    matlabbatch{1}.spm.tools.tsss.momentspace.D = {fullfile(D)};
                    matlabbatch{1}.spm.tools.tsss.momentspace.condthresh = 80;       

                    spm_jobman('run', matlabbatch);
                end
                D = reload(D);
%% External covariance matrix 
%                 if ~isequal(par.cov_meth, 'spm_sample_cov')
%                     datacovf=[fname(1:end-4) '_' par.cov_meth '_data_cov.fif'];
%                     noisecovf=[fname(1:end-4) '_' par.cov_meth '_noise_cov.fif'];
%                     if ~exist('FIFF', 'var'), global FIFF; end
%                     if isempty(FIFF), FIFF = fiff_define_constants(); end
%                     [ fid, tree ] = fiff_open(datacovf);
%                     node = fiff_dir_tree_find(tree,FIFF.FIFFB_MNE_COV);
%                     datacov = mne_read_cov(fid, node, FIFF.FIFFV_MNE_NOISE_COV);
%                     datacov.cov_meth = par.cov_meth;
%                     % [ fid, tree ] = fiff_open(noisecovf);
%                     % node = fiff_dir_tree_find(tree,FIFF.FIFFB_MNE_COV);
%                     % noisecov = mne_read_cov(fid, node, FIFF.FIFFV_MNE_NOISE_COV);
%                     % noisecov.cov_meth = par.cov_meth;
%                 else
%                     datacov = [];
%                 end

                %%
                %SNR = snr(evokedpst.avg', evokedpre.avg');

                %Define SNR and reg value (fix it with only gradiometer)
                varpst = var(evokedpst.avg');
                %varpre = var(avgpre.avg');
                ch_idx = find(varpst>=max(varpst)*0.50);
                %idxpre = find(varpre>=max(varpre)*0.75);
                %ch_name = evokedpst.label(ch_idx);
                %SNR = snr(avgpst.avg', avgpre.avg')
                SNR2 = 10*log10(snr(evokedpst.avg(ch_idx,:)', evokedpre.avg(ch_idx,:)'));
                reg2 = eval('SNR2^4/500');
                reg_form = 'SNR^4/500';
                %par.reg = eval(par.reg_form);
                %par.reg = 15
                
                %% Model head and coregister  
                clear matlabbatch
                matlabbatch{1}.spm.meeg.source.headmodel.D = {fullfile(D)};
                matlabbatch{1}.spm.meeg.source.headmodel.val = 1;
                matlabbatch{1}.spm.meeg.source.headmodel.comment = 'comments';
                matlabbatch{1}.spm.meeg.source.headmodel.meshing.meshes.mri = {mrifname};
                matlabbatch{1}.spm.meeg.source.headmodel.meshing.meshres = 2;
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).fidname = 'Nasion';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).specification.select = 'nas';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).fidname = 'LPA';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).specification.select = 'lpa';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).fidname = 'RPA';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).specification.select = 'rpa';
                matlabbatch{1}.spm.meeg.source.headmodel.coregistration.coregspecify.useheadshape = 1;
                matlabbatch{1}.spm.meeg.source.headmodel.forward.eeg = 'EEG BEM';
                matlabbatch{1}.spm.meeg.source.headmodel.forward.meg = 'Single Shell';
                spm_jobman('run', matlabbatch);
                
                hold on; scatter3(D.fiducials.pnt(:,1)/1000, D.fiducials.pnt(:,2)/1000,...
                                  D.fiducials.pnt(:,3)/1000, 100, 'ro', 'filled')
                hold on; scatter3(BF.data.MEG.sens.chanpos(:,1), BF.data.MEG.sens.chanpos(:,2),...
                                  BF.data.MEG.sens.chanpos(:,3), 2000, 'bo', 'filled')
%                 hold on; ft_plot_sens(ft_convert_units(BF.data.MEG.sens, 'm'), 'coil', true,...
%                                     'coildiameter', 0.010, 'style', 'go')
%% Apply beamforming
                SNRs = [snr_def] %, snr_mne]%, SNR_physio, SNR_golden, SNR_golden2, snr_mne_10log10];
                cnt_ = 0;
                for SNR = SNRs%reg=regs
                    cnt_ = cnt_ + 1;
                    if cnt_==1,reg = 1;
                    elseif cnt_==2, reg = 5; 
                    else reg = eval(reg_form); 
                    end  
                    reg = 5;
                    fprintf('Using. . . . . . . SNR=%f  &  Reg.=%f\n',SNR, reg)
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% temporary section
                    if cnt_==1
                        par.cov_meth= 'shrinkage';
                    elseif cnt_==2
                        par.cov_meth= 'spm_sample_cov';
                    end 
                    par.cov_meth= 'spm_sample_cov';
                    
                    if ~isequal(par.cov_meth, 'spm_sample_cov')
                        datacovf=[fname(1:end-4) '_' par.cov_meth '_data_cov.fif'];
                        noisecovf=[fname(1:end-4) '_' par.cov_meth '_noise_cov.fif'];
                        if ~exist('FIFF', 'var'), global FIFF; end
                        if isempty(FIFF), FIFF = fiff_define_constants(); end
                        [ fid, tree ] = fiff_open(datacovf);
                        node = fiff_dir_tree_find(tree,FIFF.FIFFB_MNE_COV);
                        datacov = mne_read_cov(fid, node, FIFF.FIFFV_MNE_NOISE_COV);
                        datacov.cov_meth = par.cov_meth;
                        % [ fid, tree ] = fiff_open(noisecovf);
                        % node = fiff_dir_tree_find(tree,FIFF.FIFFB_MNE_COV);
                        % noisecov = mne_read_cov(fid, node, FIFF.FIFFV_MNE_NOISE_COV);
                        % noisecov.cov_meth = par.cov_meth;
                    else
                        datacov = [];
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                    out_tmp_dir=[out_path1 'SPM//BF_' dfname '-chan_' meg{1} '-sss' num2str(sss) '-runtry' num2str(par.runtry)];
                    mkdir(out_tmp_dir)
                    cd(out_tmp_dir)

                    if isequal(par.apply_lcmv, 'yes') % Apply LCMV
                        disp('Running LCMV ----------------------->')
                        if cnt_==1;%reg == regs(1)
                            clear matlabbatch;
                            % EN_SPM_run_LCMV_v2(D, par.gridres, par.bpfreq, reg, woi, par.meg, find(not(cellfun('isempty', strfind(par.meg, char(meg))))))
                            EN_SPM_run_LCMV_v2__test(datacov,out_tmp_dir, D, par.gridres, par.bpfreq,...
                                                    reg, woi, par.meg, find(not(cellfun('isempty',...
                                                    strfind(par.meg, char(meg))))), 'cov')
                        else
                            load([out_tmp_dir '/BF_LCMV.mat'])
                            unix(['rm -f ' [out_tmp_dir '/BF_LCMV.mat']])
                            save([out_tmp_dir '/BF.mat'], 'data', 'sources')
                            clear data source features inverse output  
                            clear matlabbatch
                            % EN_SPM_run_LCMV2_v2(out_tmp_dir, par.bpfreq, reg, woi, par.meg, find(not(cellfun('isempty', strfind(par.meg, char(meg))))))
                            EN_SPM_run_LCMV2_v2__test(datacov, out_tmp_dir, par.bpfreq, reg, woi,...
                                                      par.meg, find(not(cellfun('isempty', ...
                                                      strfind(par.meg, char(meg))))))
                        end

                        % find hotspot, peak value and localization error
                        movefile('BF.mat', 'BF_LCMV.mat')
                        BF = load('BF_LCMV.mat');
                        % %
                        [~, hind]=max(abs(BF.output.image.val));
                        hval = BF.output.image.val(hind);
                        hspot = BF.sources.pos(hind, :)*1000;
                        difff = sqrt(sum((act_loc-hspot).^2));
                        disp([hspot, hval, difff])

                        % % Calculate the depth from bem surface
                        hspot_m   = hspot/1000;
                        act_loc_m = act_loc/1000;
                        [pntidx, pntdist] = dsearchn( BF.data.MEG.vol.bnd.pnt, hspot_m);
                        [pntidx_, pntdist_] = dsearchn( BF.data.MEG.vol.bnd.pnt, act_loc_m);
                        closestpnt =  BF.data.MEG.vol.bnd.pnt(pntidx,:);
                  
                        figure()
                        ft_plot_vol(BF.data.MEG.vol, 'facecolor', 'none', 'edgecolor', 'y'), hold on
                        scatter3(act_loc_m(:,1), act_loc_m(:,2), act_loc_m(:,3), 100, '^g', 'filled'), hold on
                        scatter3(hspot_m(:,1), hspot_m(:,2), hspot_m(:,3), 100, '^r', 'filled')
                        
                        % find number of active sources and PSVolume >>>>>>>>>>>>>>>>>>
                        val = abs(BF.output.image.val)';
                        clust_data1 = val(val > max(val(:))*0.50)*1.0e+20;
                        n_act_grid = length(val(val > max(val(:))*0.50));
                        PSVol = n_act_grid*(par.gridres^3);
%                         fprintf('Act. Location = [%.2f, %.2f, %.2f]mm\n', act_loc(1), act_loc(2), act_loc(3))
%                         fprintf('Est. Location = [%.2f, %.2f, %.2f]mm\n', hspot(1), hspot(2), hspot(3)) 
%                         fprintf('Localization Error = %.2f \nValue at Est_loc = %.2f \n', difff, hval)
%                         fprintf('No. of active sources = %d \nPoint Spread Volume = %dmm3',n_act_grid, PSVol)
                         
                        % find 50 peak source points and plot stc
                        if isequal(par.more_plots, 'sel') && difff>10
                            clear STC
                            [valx, indx] = sort(abs(val),'descend');
                            n_hind = indx(1:50)';
                            cnt = 0;
                            for ii=n_hind
                                cnt=cnt+1;
                                STC(:,cnt)     = (BF.inverse.MEG.W{1,ii}*evoked.avg)';
                                STC_var(:,cnt) = var(STC(:,cnt));
                            end

                            if isequal(par.more_plots, 'sel')
                                %%Fs=1000; Fbp = [1 45]; stc = ft_preproc_bandpassfilter((STC'), Fs, Fbp);
                                stc = evoked;
                                stc.avg =  abs(STC');
                                stc.var = [];
                                stc.cov = cov(STC);
                                labels = regexp(num2str(1:50), ' ', 'split');
                                labels = labels(~cellfun('isempty', labels));
                                stc.label =  strcat('VERT', labels)';
                                stc.grad = [];
                                stc.grad.label = stc.label; 
                                stc.grad.chanpos =BF.sources.pos(n_hind,:);
                                cfg.viewmode = 'vertical';
                                %%%%% ft_databrowser([],stc)
                                cfg           = [];
                                cfg.output    = 'pow'; 
                                cfg.method    = 'mtmconvol';
                                cfg.taper     = 'hanning';
                                cfg.toi       = par.ctrlwin(1):0.005:par.actiwin(2); 
                                cfg.foi       = par.bpfreq(1):2:par.bpfreq(2);
                                cfg.t_ftimwin = ones(size(cfg.foi)) * 0.005;
                                TFR     = ft_freqanalysis(cfg, stc);

                                figure('name', dfname)
                                subplot_tight(2,4,[1 3], [0.07 0.005]), plot(evoked.time, STC), 
                                title([dfname '> STC plot (max 50) [Loc. Error=' num2str(difff) ']'], 'fontsize',fs, 'Interpreter', 'none')
                                subplot_tight(2,4,4, [0.07 0.005]);
                                powspctrm=squeeze(mean(TFR.powspctrm,1));
                                ft_singleplotTFR([], TFR), title('Power spectrum plot for peak STC', 'FontSize', fs)
                                colorbar('South'),  colorbar off 
                                subplot_tight(2,4,[5,6], [0.07 0.005]), plot(evoked.time, abs(STC)), 
                                title('STC plot from max 50 sources (absolute values)', 'FontSize', fs)
                                subplot_tight(2,4,7, [0.07 0.005]), imagesc(cov(STC(1:200,:))), 
                                title(['Noise Cov [' num2str(min(min(cov(STC(1:200,:))))) ' ' num2str(max(max(cov(STC(1:200,:))))) ']'], 'FontSize', fs)
                                subplot_tight(2,4,8, [0.07 0.005]), imagesc(cov(STC(201:400,:))), 
                                title(['Data Cov [' num2str(min(min(cov(STC(201:400,:))))) ' ' num2str(max(max(cov(STC(201:400,:))))) ']'], 'FontSize', fs)
                                set(gcf, 'Position', get(0, 'Screensize'));
                                try
                                    %save2word(result_figfile); close()
                                    img = getframe(gcf);
                                    imwrite(img.cdata, [data_path 'SPM//' dfname '-STC.jpeg']), clear img, close()
                                catch err
                                    disp(err), close()                                    
                                end
                            end
                        end
                    %% Cluster and plot blobs
                        clust_data  = val;
                        clust_data(val > max(val(:))*0.50) = clust_data1;
                        
                        clear idx Cent sund Dist;
                        while ~exist('idx', 'var')
                            try
                                [idx,Cent,sumd,Dist] = kmeans(clust_data,2);
                            catch err
                            end
                        end
                        % % [idx,Cent,sumd,Dist] = kmeans(clust_data,2);
                        idx(isnan(idx)) = 0;
                        if length(idx(idx==1))>length(idx(idx==2)), xxx=2; else xxx=1; end
                        % figure(), plot(idx, 'Color','b'), hold on, plot(idx==xxx, 'Color','r')
                        locs1 = BF.sources.pos(idx==xxx,:); 
                        cntrd_locs1 = [mean(locs1(:,1)),mean(locs1(:,2)),mean(locs1(:,3))];
                        n_act_vert  = size(locs1,1);
                        totalVolume2 = n_act_vert*(par.gridres^3);
                        loc_err2= sqrt(sum((act_loc-cntrd_locs1*1000).^2));
                        clear kk
                        for ii=1:n_act_vert
                            kk(ii) = sqrt(sum((locs1(ii,:)-cntrd_locs1).^2));
                        end
                        meandist = (sum(kk)/n_act_vert)*1000;
                        try
                            while size(locs1,1)<4
                                locs1 = [locs1;locs1(end,:)*1.1];
                                locs1 = [locs1;locs1(end,:)+0.001];
                            end
                            tris1 = convhull(locs1(:,1),locs1(:,2),locs1(:,3));
                            [totalVolume,totalArea] = stlVolume((locs1*1000)', tris1'); % this volume is not much reliable
                        catch err
                            totalVolume = 0;
                            totalArea   = 0;
                            if n_act_vert>2, tris1= nchoosek(1:n_act_vert,3); end
                        end
                        
                        if isequal(par.more_plots, 'sel')
                            % locs2 = source_int_mm.pos(idx==2,:); 
                            % outside = source_int_mm.pos(idx==0,:); 
                            messagee = sprintf([char(dfname) ', No. of active voxels = ' num2str(n_act_grid), ...
                                                '\nVolume enclosed = ' num2str(totalVolume) 'mm3', ...
                                                ', Point spread volume = ' num2str(PSVol) 'mm3', ...
                                                ', Regularization Value = ' num2str(reg),...
                                                ', LocErr=' num2str(difff)]);
                            figure()
                            annotation('textbox',[.01 .85 .98 .15], 'LineWidth', 0.0001,'FontSize', 11,...
                                        'String',messagee, 'Interpreter', 'none')
                            vw = [0,90;-90,0;0,0];
                            for ii=1:3
                                subplot_tight(1,3,ii,0.0), 
                                scatter3(locs1(:,1),locs1(:,2),locs1(:,3), 'MarkerFaceColor',[0 .7 .7]), hold on
                                scatter3(cntrd_locs1(:,1),cntrd_locs1(:,2),cntrd_locs1(:,3), 'MarkerFaceColor','red'), hold on
                                trisurf(tris1, locs1(:,1),locs1(:,2),locs1(:,3), 'FaceAlpha', 0.9), camlight, hold on
                                % scatter3(locs2(:,1), locs2(:,2), locs2(:,3), 'MarkerFaceColor',[.7 .7 0]), hold on
                                % scatter3(outside(:,1),outside(:,2),outside(:,3)), hold on
                                ft_plot_vol(ft_convert_units(BF.data.MEG.vol, 'm'), 'facecolor', 'None', 'edgecolor', 'b'), camlight, alpha 0.3
                                % trisurf(convhull(kk), kk(:,1),kk(:,2),kk(:,3), 'edgecolor', 'none' ), alpha 0.2
                                view(vw(ii,:)), %rotate3d, axis off
                            end
                            set(gcf, 'Position', get(0, 'Screensize')); 
                            img = getframe(gcf);
                            imwrite(img.cdata, [data_path 'SPM//' dfname '-blob.jpeg']), clear img; close()
                            % saveas(gcf, [data_path 'SPM//' dfname '-blob.jpeg'], 'jpg'), close
                        end

                        fprintf('########################################################\n')
                        fprintf('Actual source Location \t = [%.1f, %.1f, %.1f]mm\n', loc)
                        fprintf('Estimated source Location= [%.1f, %.1f, %.1f]mm\n', hspot)
                        fprintf('Localization Error \t = %.1fmm\n', difff)
                        fprintf('No. of active sources \t = %d \nPoint Spread Volume(PSV) = %dmm3\n',n_act_grid, PSVol)
                        fprintf('Total envelop volume \t = %.1fmm3\n', totalVolume)
                        fprintf('Total envelop area \t = %.1fmm2\n', totalArea)
                        fprintf('########################################################\n')
            
                        % Write results in a text file
                        resultfile=[out_path 'SPM_BF_Simulations_Est_SourceLoc__1.csv'];
                        if linenum==pick_datalines(1) && cnt_==1 %&&  reg == regs(1)
                            fid = fopen(strcat(resultfile), 'a+');
                            fprintf(fid, '**************************************\nDate&time=%s\n', datestr(now));
                            fprintf(fid, 'Gridres=%.1f,\n', par.gridres);
                            fprintf(fid, 'Bandpass=%.1f-%.1f,\n', par.bpfreq);
                            fprintf(fid, 'reg_form=%s,\n', par.reg_form);
                            fclose(fid); 
                            msg={'data','Amp','x','y','z','value','LocErr','ActDist','EstDist',...
                                'Ntrials','Nchannels','SNR','Reg','Cov_rank',' ','Method','EnvVol',...
                                'EnvArea','N_ActSource','PSVol','LocErr2','Cntrdx','Cntrdy','Cntrdz',...
                                'SNR2', 'mean_dist', 'noisecov_rank', 'datacov_rank', 'actdist_pnt2surf',...
                                'estdist_pnt2surf', 'vertnum', 'cov_meth','linenum'};
                            fid = fopen(strcat(resultfile), 'a+');
                            
                            for ii=msg
                                fprintf(fid, '%s,', char(ii));
                            end
                            fprintf(fid, '\n');
                            fclose(fid);
                        end
                        fid = fopen(strcat(resultfile), 'a+');
                        fprintf(fid, '%s,',   dfname);
                        fprintf(fid, '%d,',   amp);    
                        fprintf(fid, '%.2f,', [hspot(1),hspot(2),hspot(3), hval, difff]);%, );
                        fprintf(fid, '%.2f,', sqrt(sum(([0,0,0]-act_loc).^2)));
                        fprintf(fid, '%.2f,', sqrt(sum(([0,0,0]-hspot).^2)));
                        fprintf(fid, '%d,',   [D.ntrials, D.nchannels]);
                        fprintf(fid, '%f,',   SNR);
                        fprintf(fid, '%f,',   reg);
                        fprintf(fid, '%d,',   rank(BF.features.MEG.Cinv, 1e-20)); %rank
                        fprintf(fid, '%s,',   '');
                        fprintf(fid, '%s,',   'LCMV');
                        fprintf(fid, '%.2f,', totalVolume);
                        fprintf(fid, '%.2f,', totalArea);
                        fprintf(fid, '%d,',   n_act_grid);
                        fprintf(fid, '%.2f,', PSVol);
                        fprintf(fid, '%.2f,', loc_err2);
                        fprintf(fid, '%.2f,', [cntrd_locs1(1),cntrd_locs1(2),cntrd_locs1(3)]*1000);
                        fprintf(fid, '%f,',   SNR2);
                        fprintf(fid, '%.1f,', meandist);
                        fprintf(fid, '%d,',   noisecov_rank);
                        fprintf(fid, '%d,',   datacov_rank);
                        fprintf(fid, '%.1f,', pntdist_*1000);%, );
                        fprintf(fid, '%.1f,', pntdist*1000);%, );
                        fprintf(fid, '%d,',   vertnum);
                        fprintf(fid, '%s,',   par.cov_meth);
                        fprintf(fid, '%d,',   linenum);
                        fprintf(fid, '\n');
                        fclose(fid);
                        
                        %% Plot source ortho plot
                        if isequal(par.source_ortho, 'yes')
                            M1=[];
                            M1.time     = [];
                            M1.dim      = BF.sources.grid.dim;
                            kk          = false(31320,1);
                            kk(BF.sources.grid.inside,1)=1;
                            M1.inside   = kk;
                            M1.pos      = BF.sources.grid.allpos;
                            M1.method   = 'average';
                            kk          = nan(31320,1);
                            kk(BF.sources.grid.inside) = BF.output.image.val';
                            M1.avg.pow  = kk;
                            M1.unit     =  'm'; 
                            M1.avg.mom  = [];
                            M1 = ft_convert_units(M1, 'mm');
                            
                            % Interporate NIA with MRI (Interpolate overlay on source space)
                            segmrifname = '/data/beamformer_data/Beamformer_share/multimodal/sub1/MRI/nenonen_jukka_1_01-brain.mat';
                            if ~exist('segmri', 'var'); load(segmrifname); end
                            cfg              = [];
                            cfg.parameter    = 'pow';%'avg.pow';
                            cfg.downsample   = 3; 
                            cfg.interpmethod = 'nearest';
                            source_int  = ft_sourceinterpolate(cfg, M1, segmri);
                            source_int = ft_convert_units(source_int, 'mm');
                            source_int.pow = source_int.avg.pow(:);
                            source_int.mask = source_int.pow> max(source_int.pow(:))*0.50; % Set threshold for plotting
                            cfg                 = [];
                            cfg.method          = 'ortho';
                            cfg.funparameter    = 'pow';%'avg.pow';
                            cfg.maskparameter   = 'mask';
                            cfg.funcolormap     = 'hot';
                            cfg.colorbar        = 'yes';
                            %cfg.location        = hspot;
                            ft_sourceplot(cfg, source_int);
                            camroll(180)
%                             set(gcf,'Position',[100 100 2000 1600])
%                             set(gcf, 'PaperSize', [8 2]);
                            set(gcf, 'Position', get(0, 'Screensize'));
                            
                            messagee = sprintf([char(dfname) ', \nNo. of active voxels=' num2str(n_act_vert), ...
                                        ',  Volume enclosed=' num2str(totalVolume) 'mm3', ...
                                        ',\nPoint spread volume=' num2str(PSVol) 'mm3', ...
                                        ',  Regularization Value=' num2str(reg), ...
                                        ',\nPeak location(in grid)= [' num2str(hspot) ']mm', ...
                                        ',  Peak Value=' num2str(hval), ...
                                        '\nCentroid location=[' num2str(cntrd_locs1*1000) ']mm',...
                                        '  Loc. Error=' num2str(difff) 'mm']);
                            annotation('textbox',[.45 .06 .51 .18], 'LineWidth', 0.0001,...%'FontSize', 9,...
                                        'String',messagee, 'BackgroundColor', [1 1 1], 'Interpreter', 'none')
                            saveas(gcf, [data_path 'SPM//' dfname '-source.jpeg'], 'jpg'),  close
                        end
                    end

                    if isequal(par.apply_dics, 'yes')  % Apply DICS
                        clear matlabbatch
                        disp('Running DICS ----------------------->')
                        EN_SPM_run_DICS(D, par.gridres, reg, woi, dics_freq, par.meg, meg)

                        % Write hotspot (dipole) location in a text file
                        movefile('BF.mat', 'BF_DICS.mat')
                        BF = load('BF_DICS.mat');
                        %[hval, hind]=max(BF.output.image.val);
                        [~, hind]=max(abs(BF.output.image.val));
                        hval = BF.output.image.val(hind);
                        hspot=BF.sources.pos(hind, :)*1000;
                        difff=sqrt(sum((actual_diploc(dip,:)-hspot).^2));
                        disp([hspot, hval, difff]);
                        
                        % find the focal length >>>>>>>>>>>>>>>>>>
                        val = abs(BF.output.image.val)';
                        clust_data1 = val(val > max(val(:))*0.50)*1.0e+20;
                        clust_data  = val;
                        clust_data(val > max(val(:))*0.50) = clust_data1; 

                        [idx,C,sumd,D] = kmeans(clust_data,2);
                        idx(isnan(idx))=0;
                        % figure(), plot(idx, 'Color','b'), hold on, plot(idx==1, 'Color','r')
                        locs1 = BF.sources.pos(idx==1,:);
                        cntrd_locs1 = [mean(locs1(:,1)),mean(locs1(:,2)),mean(locs1(:,3))];
                        n_act_vert = size(locs1,1);
                        totalVolume2 = n_act_vert*(par.gridres^3);
                        try
                            tris1 = convhull(locs1(:,1),locs1(:,2),locs1(:,3));
                            [totalVolume,totalArea] = stlVolume((locs1*1000)', tris1'); % this volume is not much reliable
                        catch err
                            totalVolume = 0;
                            totalArea   = 0;
                            if n_act_vert>2, tris1= nchoosek(1:n_act_vert,3); end
                        end
                        if isequal(par.more_plots, 'yes')
                            locs2   = BF.sources.pos(idx==2,:); 
                            figure()
                            scatter3(locs1(:,1),locs1(:,2),locs1(:,3), 'MarkerFaceColor',[0 .7 .7]), hold on
                            scatter3(cntrd_locs1(:,1),cntrd_locs1(:,2),cntrd_locs1(:,3), 'MarkerFaceColor','red'), hold on
                            trisurf(tris1, locs1(:,1),locs1(:,2),locs1(:,3), 'FaceAlpha', 0.9), camlight, hold on
                            scatter3(locs2(:,1), locs2(:,2), locs2(:,3), 'MarkerFaceColor',[.7 .7 0]), hold on
                            ft_plot_vol(ft_convert_units(BF.data.MEG.vol, 'm'), 'facecolor', 'none', 'edgecolor', 'b'), camlight, alpha 0.3
                            rotate3d
                        end
                        disp(['Total volume = ' num2str(totalVolume) ' mm3'])
                        disp(['Total volume = ' num2str(totalArea) ' mm2'])
                        disp(['Actual total volume = ' num2str(totalVolume2) ' mm3'])

                        resultfile_dics=[out_path 'SPM_BF' '-' par.site '_estimated-source_loc_DICS.csv'];
                        if meg==1 && pp==1 && sub==1 && stimval==1 && reg==regs(1)
                            fid = fopen(strcat(resultfile_dics), 'a+');
                            fprintf(fid, '\nGrid res = %s,\n', num2str(par.gridres));
                            fprintf(fid, 'Band pass= %s,\n', num2str(par.bpfreq));
                            fprintf(fid, 'reg_form = %s,\n', par.reg_form);
                            fprintf(fid, 'Date & time = %s,\n', datestr(now));
                            fprintf(fid, '\n');
                            fclose(fid);  
                        end
                        fid = fopen(strcat(resultfile_dics), 'a+');
                        fprintf(fid, '%s,', mfname);
                        fprintf(fid, '%s,', stimcat);
                        fprintf(fid, '%f,', [hspot(1),hspot(2),hspot(3), hval, difff, D.ntrials,  D.nchannels, SNR, reg]);
                        fprintf(fid, '%s,', par.prep{1,pp});
                        fprintf(fid, '%s,', 'DICS');
                        fprintf(fid, '%.2f,', totalVolume);
                        fprintf(fid, '%.2f,', totalArea);
                        fprintf(fid, '\n');
                        fclose(fid);
                    end

                    if cnt_==6% reg==regs(end)           
                        cd('~//spm_tempp//')
                        unix(['rm -rf ' out_tmp_dir])
                    else
                        cd(out_tmp_dir)
                        unix('find . -type f -name "*.nii" -delete')
                    end
                    % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    close all
                    clearex chancat datacat SSS apply_dics apply_lcmv out_tmp_dir fname act_loc...
                        reg_compare dics_freq site maxf amps dipoles iii actual_diploc homedir...
                        reg out_dir trialwin woi matlabbatch sss mfname par reg_form SNR out_path ...
                        keyset valueset evdict par channelslist homedir pick_datalines data_path out_path ...
                        mri_path par sss inputfile mrifname dfname fname linenum meg sss dfname ...
                        SNR SNR2 ignorech badch channelslist megchan keyset D regs fnamepre amp evoked fs ...
                        cnt_ SNRs noisecov_rank datacov_rank data_to_filter datacov vertnum out_path1 ...
                        leadfield headmodel % for FT leadfield & headmodel comparison
                toc;
                end
                cd('~//spm_temp//')
                unix(['rm -rf ' out_tmp_dir]) 
            end
            unix('find . -type f -name  "*spmeeg_*"  -delete')
            %unix('find . -type f -name  "*spmeeg_*"  -delete')
        end
    end
end
cd ~
toc;
%***********************************END************************************       
