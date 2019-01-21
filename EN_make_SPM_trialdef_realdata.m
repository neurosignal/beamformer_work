function [trl]= EN_make_SPM_trialdef_realdata(dataset, stimchan, stimval, trialwin, rmtrialstart, rmtrialend)
%% it's for real data
    cfg                     = [];                   
    cfg.dataset             = dataset;
    cfg.trialdef.eventtype  = stimchan;
    cfg.trialdef.eventvalue = stimval;                     
    cfg.trialdef.prestim    = abs(trialwin(1));                     
    cfg.trialdef.poststim   = trialwin(2); 
    cfg.trialfun            = 'ft_trialfun_general';
    cfg.minlength = cfg.trialdef.prestim + cfg.trialdef.poststim;
    cfg = ft_definetrial(cfg);
    % Leave the first and last trigger to avoid data deficiency error
    cfg.trl(1:rmtrialstart,:)  =[];
    if rmtrialend>0
        cfg.trl(end-rmtrialend+1:end,:)=[];
    end

    trl=[cfg.trl(:,1:2), cfg.trl(:,3)];

    fprintf(['**Eliminated first ' num2str(rmtrialstart) ' and last ' num2str(rmtrialend) ' trials to remove the reset artifacts. Resulting ' num2str(size(trl,1)) ' trials in total...........\n\n'])

end
