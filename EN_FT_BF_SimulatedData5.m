%%  FieldTrip simulated data beamforming pipeline (Combining LCMV and DICS)
%% Date: 24/07/2018
%% Author: Amit @ Elekta Neuromag
%% Add fieldtrip and spm in path
clearex leadfield headmodel%nchan_lf; 
clc; refresh
restoredefaultpath
add_ft='yes'; add_spm='no';
if ispc, homedir = 'C:\Users\fijaiami//'; 
else  homedir = '/home/amit/'; end
addpath([homedir 'git//ChildBrain//BeamComp//MATLAB//'])
EN_add_toolboxes(add_ft, add_spm);
cd(homedir)
cd('git//ChildBrain//BeamComp//MATLAB//')
close; tic;

%% Set data directory and other parameters
pick_datalines = 24385:24784;%:24784;%15051:17451;%5829:17000; % 8207:20000;
if ispc
    data_path = '\\172.16.50.206\data\rd\ChildBrain\Simulation\NEW_25_seglabBEM6\'; %NEW_DS10\';
    if exist('D:\DATA\', 'dir'), drive2write = 'D:\DATA\'; else drive2write='C:\0DATA\'; end
    out_path  =  sprintf('%sSimulations//NEW_25_seglabBEM6//FieldTrip//', drive2write);  mkdir(out_path)
    data_path1= out_path;
    % mri_path  = 'C:\Users\fijaiami\Documents\Visits&Purchases\BirmUni-Feb2018\Training\multimodal_data\MRI\';
    mri_path  = '\\172.16.50.206\data\rd\ChildBrain//neurodata//beamformer_data//Beamformer_share//multimodal//sub1//MRI//';    
elseif isunix
    data_path = '/net/qnap/data/rd/ChildBrain/Simulation/NEW_25_seglabBEM6/'; %'/net/qnap/data/rd/ChildBrain/Simulation/';
    out_path  = sprintf('%sFieldTrip//', data_path);  mkdir(out_path)
    data_path1= out_path;
    mri_path  = '/net/qnap/data/rd/ChildBrain/neurodata/beamformer_data/Beamformer_share/multimodal/sub1/MRI/';
end
resultfile= sprintf('%sgit//ChildBrain//BeamComp//BeamComp_Resultfiles//Simulations/FT_BF_Simulations_Est_SourceLoc.csv', homedir);
par                     = [];
par.prep                = {'', '_sss', '_tsss', '_tsss_mc','_cxsss', '_nosss'};
par.meg                 = {'all', 'mag', 'grad'};
par.visual              = 'no';
par.powspect            = '';
par.browse              = ''; 
par.more_plots          = '';
par.source_ortho        = 'no';
par.resultplotsave      = '';
par.plot_stc            = 'no';
par.runtry              = 1;
par.gridres             = 5.0; % in mm
par.bpfreq              = [2, 40];
par.bsfreq              = [49, 51];
par.cov_cut             = [0.1, 98];
par.zscore_cut          = [99, 0.1];
par.stimchan            = 'STI101';
par.mri_realign         = 'no';
par.mri_seg             = 'no';
par.apply_pca           = 'no';
par.reg_form            = 'SNR^4/500';
par.reg_compare         = '';
par.data_to_filter      = 'continuous';
par.use_spm_headmodel   = 'yes';
par.scale_data_fromT2fT = 'yes';
par.save_rawdata_4_MNE  = 'no';
fprintf('data_path= %s\ndata_path1= %s\nmri_path= %s\nout_path= %s\nresultfile= %s\n',...
        data_path, data_path1, mri_path, out_path, resultfile)
%% Read row-wise entries from first column from the result csv file
if ispc
    inputfile      = [homedir 'git//ChildBrain//BeamComp//BeamComp_Resultfiles//Simulations//Simulations_Est_SourceLoc_1.csv'];
    plot_errorfile = [homedir 'git//ChildBrain//BeamComp_stote//FT_BF_Simulations_Est_SourceLoc_plot_error1.txt'];
    result_figfile = [homedir 'git//ChildBrain//BeamComp_stote\FT_Simulations_Est_SourceLoc_plots18.doc'];
elseif isunix
    inputfile      = [homedir 'git//ChildBrain//BeamComp//BeamComp_Resultfiles//Simulations//Simulations_Est_SourceLoc_1.csv'];
    plot_errorfile = [homedir 'git//ChildBrain//BeamComp//BeamComp_Resultfiles//Simulations//FT_BF_Simulations_Est_SourceLoc_plot_error.txt'];
    result_figfile = [homedir 'git//ChildBrain//BeamComp//BeamComp_Resultfiles//Simulations//FT_BF_Simulations_Est_SourceLoc_Figs.doc'];
end
mrifname    = [mri_path 'nenonen_jukka-amit-170204-singlecomplete.fif'];
segmrifname = [mri_path 'nenonen_jukka_1_01-brain.mat'];
fnamepre    = ''; %just to initialize
%%
for meg = {'all'}%,'grad','mag'}
    for linenum = pick_datalines
        if ismac
            dfname = NaN;
            while ~isstr(dfname)
                [~,~,lineraw] = xlsread(inputfile, ['A' num2str(linenum) ':' 'Z' num2str(linenum)]);
                dfname = [lineraw{1,1}];
                disp('Wait ..............')
            end
            disp(dfname)
            fname = [data_path dfname '.fif'];
            loc = regexp(lineraw{1,1},'(?<=nAm_at_).*(?=mm-ave_OVER_)','match');
            loc = regexp(loc, '_', 'split');
            loc = cellfun(@str2double,loc{1,1});
            SNR = lineraw{1,12};
            amp = lineraw{1,2};
            
        elseif isunix || ispc
            endline = 1;
            while linenum>endline
                fid         = fopen(inputfile);
                track_data  = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s ','delimiter', ',', 'EmptyValue', -Inf);
                fclose(fid);
                disp('wait ...................')
                endline = length(track_data{1,1});
                pause(1)
            end
            for mm=1:size(track_data,2) % just to avoid alternate extra line
                track_data{1,mm}= track_data{1,mm}(1:2:length(track_data{1,end}));
            end
            disp('proceed >>>>>>>>>>>>>>>>>>')
            
            dfname      = track_data{1,1}{linenum,1};
            fname       = [data_path dfname '.fif'];
            act_loc     = regexp(dfname,'(?<=nAm_at_).*(?=mm)','match');
            act_loc     = regexp(act_loc, '_', 'split');
            act_loc     = cellfun(@str2double,act_loc{1,1});
            SNR         = str2double(track_data{1,12}{linenum,1});
            amp         = str2double(track_data{1,2}{linenum,1});
            loc         = act_loc;
            LocErr_     = str2double(track_data{1,7}{linenum,1});
            
            snr_def     =  SNR;
            snr_mne     = SNR%str2double(track_data{1,12}{linenum+1,1});
%             SNR_physio  = str2double(track_data{1,12}{linenum+2,1});
%             SNR_golden  = str2double(track_data{1,12}{linenum+3,1});
%             SNR_golden2 = str2double(track_data{1,12}{linenum+4,1});
%             snr_mne_10log10 = 10*log10(snr_mne);
            
            clear track_data
        end
        load([homedir 'git//ChildBrain//BeamComp/MATLAB/channelslist.mat']);
        if isequal(meg, {'all'})
            ignorech = [];
        elseif isequal(meg, {'mag'})
            ignorech = channelslist.grad;
        elseif isequal(meg, {'grad'})
            ignorech = channelslist.mag;
        end
        if isequal(fname(end-6:end-4), 'raw') || isequal(fname(end-8:end-4), 'nossp')
            bads       = {'MEG2233', 'MEG2212', 'MEG2332', 'MEG0111'}; 
        else
            bads       = {};
        end
        badch        = [ignorech, bads]; 
        par.trialwin = [-0.2 0.2];
        par.ctrlwin  = [-0.1 -0.01];
        par.actiwin  = [0.01 0.1];
        par.badtrs   = [15, 103,104,107]; % 
        megchan=channelslist.meg;
        megchan(ismember(megchan, badch))=[]; 

        % Find trigger categories && label them
        keyset = {'Simulated'};  valueset = 5;
        evdict=containers.Map(keyset, valueset);
        %%
        s_resfigfile = dir(result_figfile);
        if isempty(s_resfigfile)
                s_resfigfile_size = 1;
            else
                s_resfigfile_size = s_resfigfile.bytes;
        end
        cnt=0;
        while s_resfigfile_size>=120000000
            cnt = cnt+1;
            fnum=regexp(result_figfile, '\d*','Match');
            result_figfile = [result_figfile(1:51) num2str(str2double(cell2mat(fnum))+1) '.doc'];
            s_resfigfile = dir(result_figfile);
            if isempty(s_resfigfile)
                s_resfigfile_size = 1;
            else
                s_resfigfile_size = s_resfigfile.bytes;
            end
        end
        disp(result_figfile)
          %% 
        if ~isequal(fnamepre, fname) %% && LocErr_>10.0 && amp>100;
            disp(linenum)
            fnamepre = fname;
            disp(fname) % do something
            disp([loc, SNR])
            
            par.amp = amp;
            par.loc = loc;
            par.linenum = linenum;
            par.pick_datalines = pick_datalines;
            
            %% Browse raw data
            if isequal(par.browse,'yes')
                cfg                 = [];
                cfg.channel         = [megchan par.stimchan];
                cfg.viewmode        = 'vertical';
                cfg.blocksize       = 15;
                cfg.ylim            = [-1e-11 1e-11]; %'maxmin';
                cfg.preproc.demean  = 'yes';
                cfg.dataset         = fname;
                ft_databrowser(cfg);
            end
            %% Define trials
            cfgg                     = [];                  % empty configuration
            cfgg.dataset             = fname;               % data file name
            cfgg.channel             = megchan;
            cfgg.trialfun            = 'ft_trialfun_general';
            cfgg.trialdef.eventtype  = par.stimchan;        % trigger channel
            cfgg.trialdef.eventvalue = valueset;            % trigger value
            cfgg.trialdef.prestim    = abs(par.trialwin(1));% pre stim time in sec.
            cfgg.trialdef.poststim   = par.trialwin(2);     % post stim time in sec.
            cfgg = ft_definetrial(cfgg);          

            %% Visualize events triggers && number of trials per category
            if isequal(par.more_plots,'yes'), ft_plot_events(figure, cfgg, keyset, valueset), end
            %% For changing scale from T & T/m to fT & fT/mm (similar to SPM)
            if isequal(par.scale_data_fromT2fT, 'yes')
                clear new_chanunit
                fprintf('\nMaking new channels units for data conversion . . . . . . from T > fT and T/m > fT/mm\n')
%                 if isequal(fname(end-6:end), 'sss.fif')
                    hdr = ft_read_header(fname);
                    for ii=1:length(hdr.chanunit)
                        if isequal(hdr.chanunit{ii}, 'T')
                            new_chanunit{ii,1} = 'fT';
                        elseif isequal(hdr.chanunit{ii}, 'T/m')
                            new_chanunit{ii,1} = 'fT/mm';
                        end
                    end
%                 elseif isequal(fname(end-6:end-4), 'raw') || isequal(fname(end-8:end-4), 'nossp')
%                     for ii=1:length(megchan)
%                         if isequal(megchan{ii}(end), '1') && isequal(megchan{ii}(1:3), 'MEG')
%                             new_chanunit{ii,1} = 'fT';
%                         elseif ~isequal(megchan{ii}(end), '1') && isequal(megchan{ii}(1:3), 'MEG')
%                             new_chanunit{ii,1} = 'fT/mm';
%                         end
%                     end       
%                 end
            end
            %% Preprocess data
            if isequal(par.data_to_filter, 'continuous') % recommended
                cfg = [];
                cfg.dataset       = fname;
                cfg.channel       = 'MEG*';
                cfg.demean        = 'yes';
                cfg.bpfilter      = 'yes'; 
                %cfg.bpfiltord    = 2;
                cfg.bpfilttype    = 'but';
                cfg.bpfreq        = par.bpfreq;
                %%cfg.lpfilter     = 25;
                cfg.coilaccuracy  = 1;
                cfg.checkmaxfilter= 0;
                if isequal(par.scale_data_fromT2fT, 'yes'), cfg.chanunit = new_chanunit; end
                data = ft_preprocessing(cfg);
                if par.bpfreq(2)>45
                    cfg.bsfilter ='yes';
                    cfg.bsfreq   = [49.5 50.5];
                    data = ft_preprocessing(cfg, data);
                end
                
                cfg=[];
                cfg.channel = megchan;
                data = ft_selectdata(cfg, data);
                
                %Save data to use in MNE, if needed
                if isequal(par.save_rawdata_4_MNE, 'yes')
                    if ~isequal(par.scale_data_fromT2fT, 'yes'), hdr=ft_read_header(fname); end
                    data_ = data;
                    data_.hdr.orig = hdr.orig;
                    fieldtrip2fiff(sprintf('%sFTread_%s_bp_%d-%dHz.fif',data_path, dfname, par.bpfreq), data_); clear hdr data_
                end
                
                % % Epoch data
                cfg     = [];
                cfg.channel = megchan;
                cfg.trl = cfgg.trl(:,:); % selecttrials
                data = ft_redefinetrial(cfg, data);
                
            elseif isequal(par.data_to_filter, 'epoched') % may cause filter artifact
                cfg = cfgg;
                cfg.dataset = fname;
                cfg.channel  = megchan;
                cfg.demean     = 'yes';
                cfg.bpfilter   = 'yes'; 
                %cfg.bpfiltord  = 2;
                cfg.bpfilttype = 'but';
                cfg.bpfreq     = par.bpfreq;
                %%cfg.lpfilter       = 25;
                cfg.coilaccuracy   = 1;
                cfg.checkmaxfilter = 0;
                data = ft_preprocessing(cfg);
                if par.bpfreq(2)>45
                    cfg.bsfilter ='yes';
                    cfg.bsfreq   = [49.5 50.5];
                    data = ft_preprocessing(cfg, data);
                end
                cfg = [];
                cfg.toilim = [par.trialwin(1) + 0.020, par.trialwin(2) - 0.020];
                data = ft_redefinetrial(cfg, data);
            end
               
            %% Apply PCA
            if isequal(par.apply_pca, 'yes')
                cfg = [];
                cfg.method = 'pca';
                cfg.updatesens = 'no';
                cfg.channel = megchan;
                comp = ft_componentanalysis(cfg, data); % Decompose

                cfg = [];
                cfg.updatesens = 'no';
                cfg.component = comp.label(76:end);
                data_clean = ft_rejectcomponent(cfg, comp); % Reject beyond 75
            else
                data_clean=data;
            end
            clear data comp

            %% Interactive data browser 
            if isequal(par.more_plots, 'yes')
                cfg            = [];
                cfg.channel    = megchan;
                cfg.preproc.demean     = 'yes';
                cfg.continuous = 'no';
                cfg.viewmode   = 'butterfly';
                ft_databrowser(cfg, data_clean);
            end
            %% Bad trial and channel detection and rejection
            if isequal(par.visual, 'yes')  
                cfg          = [];
                cfg.method   = 'summary';
                cfg.metric   = 'var';
                data_summary = ft_rejectvisual(cfg,data_clean);
            else
                data_summary = data_clean;
            end
                clear('data_clean')
                
            %% Find trial variance outliers and index them 
            [selecttrials, par] = NM_ft_varcut2(data_summary, par);
            
            %% Pull SPM BEM and prepare 
            if isequal(par.use_spm_headmodel, 'yes')
                fprintf('\nUsing headmodel coregistered by SPM pipeline ......\n')
                load('/net/qnap/data/rd/ChildBrain/neurodata/beamformer_data/Beamformer_share/multimodal/sub1/MRI/nenonen_jukka-spm_generated_headmodel.mat')
                headmodel = spm_vol; clear spm_vol
                if isequal(par.scale_data_fromT2fT, 'yes') && isequal(data_summary.grad.unit, 'mm')
                   headmodel = ft_convert_units(headmodel, 'mm'); 
                end
            if ~exist('leadfield', 'var') || length(data_summary.label)~=length(leadfield.label) 
                leadfield = NM_ft_prepare_lf(par, headmodel, data_summary, mrifname , fname);
            end
            end
            %% MNE gererated headmodel
            if isequal(par.use_mne_headmodel, 'yes')
                headmodelmne = load('/net/qnap/data/rd/ChildBrain/neurodata/beamformer_data/Beamformer_share/multimodal/sub1/MRI/nenonen_jukka-mne_generated_headmodel.mat');
                headmodel=[];
                headmodel.unit = 'm';
                headmodel.bnd.coordsys = 'neuromag';
                headmodel.bnd.pos = headmodelmne.rr;
                headmodel.bnd.tri = headmodelmne.tris+1; % add 1 to chage python indexing to matlab (0->1, 1->2, ...)
                clear headmodelmne
            end
            
            %% Prepare forward model
            if ~isequal(par.use_spm_headmodel, 'yes')
                if ~exist('headmodel', 'var') || ~exist('leadfield', 'var') || length(data_summary.label)~=length(leadfield.label) % || abs(leadfield.xgrid(1))-abs(leadfield.xgrid(2))~=par.gridres/1000;
                    [headmodel, leadfield] = NM_ft_prepare_forwardmodel(par, mrifname, segmrifname, fname, data_summary);
                elseif all(ismember(data_summary.label,leadfield.label)) && all(ismember(data_summary.grad.chanunit,leadfield.cfg.grad.chanunit)) && all(ismember(data_summary.grad.unit,leadfield.cfg.grad.unit))
                    fprintf('\nUsing precomputed leadfield with same grad info...\n')
                end
            end
                                                
            %% Extract trial for this stimulus category only && average 
            cfg = [];
            % cfg.trials = find(data_summary.trialinfo(:,1) ==  valueset);
            % data = ft_preprocessing(cfg, data_summary);
            cfg.trials = selecttrials;
            data = ft_selectdata(cfg, data_summary);
            fprintf('\nRemaining #trials = %d - %d = %d trials .........\nRemoved trials: ',...
                    size(data_summary.trial,2), length(par.bad_trials), size(data.trial,2)); disp(par.bad_trials)
                
            cfg = [];
            cfg.covariance = 'yes';
            cfg.covariancewindow = par.trialwin;
            cfg.vartrllength = 2;
            evoked = ft_timelockanalysis(cfg,data); 
            
            %% Define baseline data && average 
            cfg = [];
            cfg.toilim = par.ctrlwin;
            datapre = ft_redefinetrial(cfg, data);

            cfg = [];
            cfg.covariance='yes';
            cfg.covariancewindow = 'all';
            evokedpre = ft_timelockanalysis(cfg,datapre);
            
            %% Define active data && average
            cfg = [];
            cfg.toilim = par.actiwin;
            datapst = ft_redefinetrial(cfg, data);

            cfg = [];
            cfg.covariance='yes';
            cfg.covariancewindow = 'all';
            evokedpst = ft_timelockanalysis(cfg,datapst);

%% Plot Evoked data
        if isequal(par.more_plots, 'yes')
            figure(), FS = 11;
            subplot_tight(4,4,[1,2],0.05);  
            plot(evoked.time, evoked.avg); xlim([evoked.time(1) evoked.time(end)])
            title(dfname, 'FontSize', FS, 'Interpreter', 'none')
            subplot_tight(4,4,[5,6],0.05); clear trl_var;
            for trl = 1:size(data.trial,2), trl_var(trl,:) = max(var(data.trial{1,trl}(:,:)'));  end
            scatter(1:size(data.trial,2), trl_var, 50, 'go', 'filled'); xlim([1 size(data.trial,2)]), title('Max. variance for all selected trials', 'FontSize', FS)           
            subplot_tight(2,4,3,0.05);
            imagesc(evokedpre.cov), title(['Noise Cov [' num2str(min(min(evokedpre.cov))) ' ' num2str(max(max(evokedpre.cov))) ']'], 'FontSize', FS)
            colorbar('South')
            subplot_tight(2,4,4,0.05);
            imagesc(evokedpst.cov), title(['Data Cov [' num2str(min(min(evokedpst.cov))) ' ' num2str(max(max(evokedpst.cov))) ']'], 'FontSize', FS)
            colorbar('South')
            subplot_tight(2,4,5,0.05);   
            cfg = []; cfg.layout = 'neuromag306mag.lay';
            ft_multiplotER(cfg, evoked); 
            title('Magnetometer', 'FontSize', FS)
            subplot_tight(2,4,6,0.05);   
            cfg = []; cfg.layout = 'neuromag306planar.lay';
            ft_multiplotER(cfg, evoked); 
            title('Gradioometer', 'FontSize', FS)
            cfg=[];  cfg.method = 'mtmfft'; cfg.output = 'pow';
            cfg.foi  = 1:1:25; cfg.taper = 'hanning';
            tfr= ft_freqanalysis(cfg, evokedpst);
            subplot_tight(2,4,7,0.05);
            cfg = []; cfg.layout = 'neuromag306mag.lay'; 
            ft_topoplotTFR(cfg, tfr); title('Post-stim Mags', 'FontSize', FS)
            subplot_tight(2,4,8,0.05);
            cfg = []; cfg.layout = 'neuromag306planar.lay';
            ft_topoplotTFR(cfg, tfr); title('Post-stim Grads', 'FontSize', FS), clear tfr
            set(gcf, 'Position', get(0, 'Screensize'));
        end 
%         try
% %           %save2word(result_figfile); close()
%             saveas(gcf, sprintf('%sFieldTrip//%s-chan_%s-data.jpeg', data_path, dfname, char(meg)), 'jpg'), close
%         catch err
%             disp(err), close()
%         end
%% Compute SNR 
        varpst = var(evokedpst.avg');
        ch_idx = find(varpst>=max(varpst)*0.50);
        SNR2 = 10*log10(snr(evokedpst.avg(ch_idx,:)', evokedpre.avg(ch_idx,:)'));
        par.reg = eval(par.reg_form);
        SNRs = [snr_def, SNR2];
%% Compute and apply LCMV
        par.cov_meth = 'ft_sample_cov';
        NM_ft_scan_lcmv2(out_path, fname, dfname, [snr_def], par, leadfield, headmodel,...
                        evoked, evokedpre, evokedpst, segmrifname, resultfile)
                                
        end
    end
end