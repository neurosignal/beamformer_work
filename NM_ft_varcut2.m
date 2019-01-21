%% Usage   : LCMV beamformer pipeline for neuromag dataset using FieldTrip
%% Scripted: Amit Jaiswal @ MEGIN, Helsinki, Finland 
%% Support : support@megin.fi (CC to amit.jaiswal@megin.fi)
%% Created : 15 Dec. 2018, Version: 1.0.3
%% File No : NM/FT0919C1512
%% ***********************************************************************
function [selecttrials, par] = NM_ft_varcut2(data_summary, par)
    fprintf('\nDetecting the trials with excessive max/min variance...\n')
    clear trl_var trl_zscore
    trlindx = 1:size(data_summary.trial,2);
    %timelk = D.fttimelock.trial;
     for trl = trlindx
         trl_var(trl,:)    = max(var(data_summary.trial{1,trl}(:,:)'));
         trl_zscore(trl,:) = max(max(zscore(data_summary.trial{1,trl}(:,:)')));
     end
     percentiles = prctile(trl_var, par.cov_cut);
     outlr_idx = trl_var < percentiles(1) | trl_var > percentiles(2);
     bd_trl_var = trlindx(outlr_idx);
     percentiles = prctile(trl_zscore, par.zscore_cut);
     outlr_idx = trl_zscore < percentiles(1) | trl_zscore > percentiles(2);
     bd_trl_zscore = trlindx(outlr_idx);
     bd_trls = bd_trl_var;
     % disabled zscore:     bd_trls = union(bd_trl_var, bd_trl_zscore);
     par.bad_trials = sort([par.badtrs, bd_trls]);
     if isequal(par.more_plots, 'yes')
         figure
         subplot(211),scatter(trlindx, trl_var, 25, 'bD'); xlim([trlindx(1) trlindx(end)]), title('Max. variance'), hold on
         subplot(212),scatter(trlindx, trl_zscore, 25, 'b^'); xlim([trlindx(1) trlindx(end)]), title('Max. z-score')
         subplot(211),scatter(par.bad_trials, trl_var(par.bad_trials), 40, 'ro', 'linewidth',2); xlim([trlindx(1) trlindx(end)]), title('Max. variance')
     end
     fprintf('Found total %d trials to remove ...\n',length(par.bad_trials))
     selecttrials = setdiff(trlindx, par.bad_trials);
end
