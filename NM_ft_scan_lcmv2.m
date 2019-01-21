function NM_ft_scan_lcmv2(out_path, fname, dfname, SNRs, par,leadfield, headmodel, ...
                                    evoked, evokedpre, evokedpst, segmrifname, resultfile)
    loc = par.loc;                           
    cnt1 = 0;
    for SNR = SNRs%, snr_mne, SNR_physio, SNR_golden, SNR_golden2, snr_mne_10log10]
    % for regul = [5, eval(par.reg_form)]%, eval('SNR2^4/500')]
        cnt1 = cnt1 + 1;
        if cnt1==1, regul=5; else regul=eval(par.reg_form); end
        if regul<=0.0001, regul = 0.0001;  end                
        par.reg=regul;
        fprintf('Using. . . . . . . SNR=%f  &  Reg.=%f\n',SNR, par.reg)
        %% Source analysis
        cfg                  = [];
        cfg.method           = 'lcmv';
        cfg.grid             = leadfield;
        cfg.headmodel        = headmodel; 
        cfg.lcmv.keepfilter  = 'yes';
        cfg.lcmv.fixedori    = 'yes'; % project on axis of most variance using SVD
        cfg.lcmv.reducerank  = 2;
        cfg.lcmv.normalize   = 'yes'; %%%%%%%%%%
        cfg.senstype         = 'MEG';
        cfg.lcmv.lambda      = [num2str(par.reg) '%'];
        fprintf('\n\nCreating spatial filter using leadfield and covariance with regularization %s > > > > > \n\n', cfg.lcmv.lambda)
        source_avg           = ft_sourceanalysis(cfg, evoked); % create spatial filters

        cfg                  = [];
        cfg.method           = 'lcmv';
        cfg.senstype         = 'MEG';
        % cfg.lcmv.feedback    = 'gui';
        cfg.grid             = leadfield;
        cfg.grid.filter      = source_avg.avg.filter;
        cfg.headmodel        = headmodel;
        fprintf('\n\nScanning for baseline data > > > > > \n\n')
        sourcepreM1=ft_sourceanalysis(cfg, evokedpre); % apply spatial filter on baseline data
        fprintf('\n\nScanning for active data > > > > > \n\n')
        sourcepstM1=ft_sourceanalysis(cfg, evokedpst); % apply spatial filter on active data

        %% Calculate and save Neural Activity Index (Overlay)
        fprintf('\n\nCalculating NAI (Neural Activity Index) > > > > > \n\n')
        M1          = sourcepstM1;
        M1.avg.pow  = (sourcepstM1.avg.pow-sourcepreM1.avg.pow)./sourcepreM1.avg.pow;

        if isequal(par.more_plots, 'yes')
            figure()
            scatter3(M1.pos(M1.inside,1), M1.pos(M1.inside,2), M1.pos(M1.inside,3)), hold on
            ft_plot_vol(ft_convert_units(headmodel, 'm'), 'facecolor', 'None', 'edgecolor', 'brain'), camlight, alpha 0.3
            ft_plot_headshape(ft_convert_units(ft_read_headshape(fname), 'm'), 'facecolor', 'None', 'edgecolor', 'brain'), camlight, alpha 0.3
            rotate3d
        end
        %% find hotspot (dipole) location >>>>>>>>>>>>>>> 
        M1.avg.pow(isnan(M1.avg.pow))=0;
        [~,hind] = max(abs(M1.avg.pow));
        hval = M1.avg.pow(hind);
        hspot = M1.pos(hind,:)*ft_scalingfactor(headmodel.unit,'mm');
        difff = sqrt(sum((loc-hspot).^2));
        n_act_grid = length(M1.avg.pow(M1.avg.pow > max(M1.avg.pow(:))*0.50));
        PSVol = n_act_grid*(par.gridres^3);
        c_dist = sqrt(sum(([0,0,0]-hspot).^2));
        % disp([hspot, hval, difff])
        
        fprintf('########################################################\n')
        fprintf('Act. Location \t\t= [%.1f, %.1f, %.1f]mm\n', loc)
        fprintf('Est. Location \t\t= [%.1f, %.1f, %.1f]mm\n', hspot) 
        fprintf('Localization Error \t= %.1fmm\n', difff)
        fprintf('No. of active sources \t= %d \nPoint Spread Volume(PSV)= %dmm3\n',n_act_grid, PSVol)
%         fprintf('########################################################')

        %% find 50 max points and plot stc
        if isequal(par.more_plots, 'sel') && difff>10 % plot only for high error
            clear STC
            [~, indx] = sort(abs(M1.avg.pow),'descend');
            n_hind = indx(1:50)';
            cnt = 0;
            for ii=n_hind
                cnt=cnt+1;
                STC(:,cnt) = (source_avg.avg.filter{ii,1}*evoked.avg)';
                STC_var(:,cnt) = var(STC(:,cnt));
            end
            stc = evoked;
            stc.avg =  abs(STC');
            stc.var = [];
            stc.cov = cov(STC);
            if ispc
                stc.label = strsplit(num2str(1:50))';
                stc.grad = [];
                stc.grad.label = strsplit(num2str(1:50))';
            elseif isunix
                stc.label = regexp(num2str(1:50), '  ', 'split')';
                stc.grad = [];
                stc.grad.label = regexp(num2str(1:50), '  ', 'split')';
            end
            stc.grad.chanpos = M1.pos(n_hind,:);
            %%%%% ft_databrowser([],stc)
            cfg           = [];
            cfg.output    = 'pow'; 
            cfg.method    = 'mtmconvol';
            cfg.taper     = 'hanning';
            cfg.toi       = stc.time; %par.ctrlwin(1):0.001:par.actiwin(2); 
            cfg.foi       = par.bpfreq(1):2:par.bpfreq(2);
            cfg.t_ftimwin = ones(size(cfg.foi)) * 0.005;
            TFR     = ft_freqanalysis(cfg, stc);

            figure('name', dfname)
            subplot_tight(2,4,[1 3], 0.05), plot(evoked.time, STC), title([dfname '> STC plot (max 50) [Loc. Error=' num2str(difff) ']'], 'Interpreter', 'none')
            subplot_tight(2,4,4, 0.05),
            powspctrm=squeeze(mean(TFR.powspctrm,1));
            ft_singleplotTFR([], TFR), title('Power spect plot for peak STC')
            colorbar('South'),  colorbar off 
            subplot_tight(2,4,[5,6], 0.05), plot(evoked.time, abs(STC)), title('STC plot from max 50 sources (absolute values)')
            subplot_tight(2,4,7, 0.05), imagesc(cov(STC(1:200,:))), title('Noise Cov.')
            subplot_tight(2,4,8, 0.05), imagesc(cov(STC(201:400,:))), title('Data Cov.')
            set(gcf, 'Position', get(0, 'Screensize'));
            %%Fs=1000; Fbp = [1 45]; stc = ft_preproc_bandpassfilter((STC'), Fs, Fbp);
            try
                %save2word(result_figfile); close()
                saveas(gcf, [data_path1 dfname '--STC.jpeg'], 'jpg'), close
            catch err
                disp(err), close()
            end
        end

        %% find the focal length >>>>>>>>>>>>>>>>>>
        clust_data1 = M1.avg.pow(M1.avg.pow > max(M1.avg.pow(:))*0.50)*1.0e+20;
        clust_data  = M1.avg.pow;
        clust_data(M1.avg.pow > max(M1.avg.pow(:))*0.50) = clust_data1; 

        clear idx C sumd D;                  
        while ~exist('idx','var')
            try
                 [idx,~,~,D] = kmeans(clust_data,2); % [idx,C,sumd,D] = kmeans(clust_data,2);
            catch err
            end
        end

        idx(isnan(idx))=0;
        if length(idx(idx==1))> length(idx(idx==2)), xxx=2; else xxx=1; end
        % figure(), plot(idx, 'Color','b'), hold on, plot(idx==xxx, 'Color','r')
        locs1 = M1.pos(idx==xxx,:);
        cntrd_locs1 = [mean(locs1(:,1)),mean(locs1(:,2)),mean(locs1(:,3))];
        %n_act_vert = size(locs1,1);
        n_act_vert = size(clust_data1,1);
        totalVolume2 = n_act_vert*(par.gridres^3);
        clear kk
        for ii=1:n_act_vert
            kk(ii) = sqrt(sum((locs1(ii,:)-cntrd_locs1).^2));
        end
        meandist = (sum(kk)/n_act_vert)*ft_scalingfactor(headmodel.unit,'mm');

        try
            clear tris1
            tris1 = convhull(locs1(:,1),locs1(:,2),locs1(:,3));
            [totalVolume,totalArea] = stlVolume((locs1*ft_scalingfactor(headmodel.unit,'mm'))', tris1'); % this volume is not much reliable
        catch err
            totalVolume = 0;
            totalArea   = 0;
            if n_act_vert>2, tris1= nchoosek(1:n_act_vert,3); end
        end

        if isequal(par.more_plots, 'sel') && difff>10 % plot only for high error
            % locs2 = source_int_mm.pos(idx==2,:); 
            % outside = source_int_mm.pos(idx==0,:); 
            messagee = sprintf([char(dfname) ', No. of active voxels = ' num2str(n_act_vert), ...
                                '\nVolume enclosed = ' num2str(totalVolume) 'mm3', ...
                                ', Volume of active voxels = ' num2str(totalVolume2) 'mm3', ...
                                '\nReg. Value = ' num2str(par.reg),...
                                ', LocErr = ' num2str(difff) 'mm, C_dist = ' num2str(c_dist) 'mm',...
                                '\nEst. Loc.= [' num2str(hspot) ']mm', ', Mean dist.= ' num2str(meandist) 'mm']);

            try
                figure()
                annotation('textbox',[.05 .82 .94 .18], 'LineWidth', 0.0001,'FontSize', 15,...
                            'String',messagee, 'Interpreter', 'none')
                vw = [0,90;-90,0;0,0];
                for ii=1:3
                    subplot_tight(1,3,ii,0.0), 
                    scatter3(locs1(:,1),locs1(:,2),locs1(:,3), 'MarkerFaceColor',[0 .7 .7]), hold on
                    scatter3(cntrd_locs1(:,1),cntrd_locs1(:,2),cntrd_locs1(:,3), 'MarkerFaceColor','red'), hold on
                    trisurf(tris1, locs1(:,1),locs1(:,2),locs1(:,3), 'FaceAlpha', 0.9), camlight, hold on
                    % scatter3(locs2(:,1), locs2(:,2), locs2(:,3), 'MarkerFaceColor',[.7 .7 0]), hold on
                    % scatter3(outside(:,1),outside(:,2),outside(:,3)), hold on
                    ft_plot_vol(ft_convert_units(headmodel2, 'm'), 'facecolor', 'None', 'edgecolor', 'brain'), camlight, alpha 0.3
                    % trisurf(convhull(kk), kk(:,1),kk(:,2),kk(:,3), 'edgecolor', 'none' ), alpha 0.2
                    view(vw(ii,:)), %rotate3d, axis off
                end
                set(gcf, 'Position', get(0, 'Screensize'));                 %%
                %save2word(result_figfile); close()
                saveas(gcf, [data_path1 dfname '--blob.jpeg'], 'jpg'), close
            catch err
                disp(err), close()
                figure()
                annotation('textbox',[.01 .85 .98 .15], 'LineWidth', 0.0001,'FontSize', 15,...
                            'String',messagee, 'Interpreter', 'none')
                vw = [0,90;-90,0;0,0];
                for ii=1:3
                    subplot_tight(1,3,ii,0.0), 
                    scatter3(locs1(:,1),locs1(:,2),locs1(:,3), 'MarkerFaceColor',[0 .7 .7]), hold on
                    scatter3(cntrd_locs1(:,1),cntrd_locs1(:,2),cntrd_locs1(:,3), 'MarkerFaceColor','red'), hold on
                    scatter3(headmodel2.bnd.pos(1:5:end,1),headmodel2.bnd.pos(1:5:end,2),headmodel2.bnd.pos(1:5:end,3), 'MarkerFaceColor','g')
    %                         ft_plot_vol(ft_convert_units(headmodel2, 'm'), 'facecolor', 'None', 'edgecolor', 'brain'), camlight, alpha 0.3
                    view(vw(ii,:)), axis off%rotate3d, 
                end
                close()
                    disp('cant print this')
            end
        end
        fprintf('Total envelop volume \t= %.1fmm3\n', totalVolume)
        fprintf('Total envelop area \t= %.1fmm2\n', totalArea)
%         fprintf('No. of active sources \t= %d' n_act_grid)
%         fprintf('Actual total volume \t= %.1fmm3' num2str(PSVol) ' mm3')
        fprintf('########################################################\n')

        %% print to file >>>>>>>>>>>>>>>>>>>>>
        if par.reg==5 && par.linenum==par.pick_datalines(1)
            msg={'data','Amp','x','y','z','value','LocErr','ActDist','EstDist',...
                'Ntrials','Nchannels','SNR','Reg',' ',' ','Method','EnvVol',...
                'EnvArea','N_ActSource','PSVol','LocErr2','Cntrdx','Cntrdy','Cntrdz', 'mean_dist', 'linenum'};
            fid = fopen(strcat(resultfile), 'a+');
            fprintf(fid, '**********\n**********\n%s\n', datestr(now));
            for ii=msg
                fprintf(fid, '%s,', char(ii));
            end
            fprintf(fid, '\n');
            fclose(fid);
        end
        fid = fopen(strcat(resultfile), 'a+');
        fprintf(fid, '%s,', dfname);
        fprintf(fid, '%d,', par.amp);    
        fprintf(fid, '%.2f,', [hspot, hval, difff]);%, );
        fprintf(fid, '%.2f,', sqrt(sum(([0,0,0]-loc).^2)));
        fprintf(fid, '%.2f,', sqrt(sum(([0,0,0]-hspot).^2)));
        fprintf(fid, '%d,', [size(evokedpst.cfg.previous.toilim,1), size(evokedpst.avg,1)]);
        fprintf(fid, '%.2f,', SNR);
        fprintf(fid, '%f,', par.reg);
        fprintf(fid, '%s,', ''); %rank
        fprintf(fid, '%s,', '');
        fprintf(fid, '%s,', 'LCMV');
        fprintf(fid, '%.2f,', totalVolume);
        fprintf(fid, '%.2f,', totalArea);
        fprintf(fid, '%d,', n_act_grid);
        fprintf(fid, '%.2f,', PSVol);
        fprintf(fid, '%.2f,', sqrt(sum((cntrd_locs1*ft_scalingfactor(headmodel.unit,'mm')-hspot).^2)));
        fprintf(fid, '%.2f,', cntrd_locs1*ft_scalingfactor(headmodel.unit,'mm'));%, );
        fprintf(fid, '%.2f,', meandist);%, );  
        fprintf(fid, '%d,',   par.linenum);%, ); 
        fprintf(fid, '%s,',  par.cov_meth);
        %%fprintf(fid, '%d,',   vertnum);%, );
        fprintf(fid, '\n');
        fclose(fid);
    %% Plot result 
    if isequal(par.source_ortho, 'yes') && difff>10 % plot only for high error
        % Interporate NIA with MRI (Interpolate overlay on source space)
        if ~exist('segmri', 'var'); load(segmrifname); end
        cfg              = [];
        cfg.parameter    = 'pow';%'avg.pow';
        cfg.downsample   = 3; 
        cfg.interpmethod = 'nearest';
        source_int  = ft_sourceinterpolate(cfg, M1, segmri);
        source_int = ft_convert_units(source_int, 'mm');
        source_int.mask = source_int.pow > max(source_int.pow(:))*0.50; % Set threshold for plotting
        cfg                 = [];
        cfg.method          = 'ortho';
        cfg.funparameter    = 'pow'; %'avg.pow';
        cfg.maskparameter   = 'mask';
        cfg.funcolormap     = 'hot';
        cfg.colorbar        = 'yes';
        %cfg.location        = hspot;
        ft_sourceplot(cfg, source_int);
        camroll(180)
        set(gcf, 'Position', get(0, 'Screensize'));
    %             messagee = sprintf([char(dfname) ', \nNo. of active voxels=' num2str(n_act_vert), ...
    %                                     ',  Volume enclosed=' num2str(totalVolume) 'mm3', ...
    %                                     ',\nVolume of active voxels=' num2str(PSVol) 'mm3', ...
    %                                     ',  Regularization Value=' num2str(par.reg), ...
    %                                     ',\nPeak location(in grid)= [' num2str(hspot) ']mm', ...
    %                                     ',  Peak Value=' num2str(hval), ...
    %                                     '\nCentroid location=[' num2str(cntrd_locs1*ft_scalingfactor(headmodel.unit,'mm')) ']mm',...
    %                                     '  Loc. Error=' num2str(difff) 'mm']);
        messagee = sprintf([char(dfname) ', No. of active voxels=' num2str(n_act_vert), ...
                                ', Volume enclosed=' num2str(totalVolume) 'mm3', ', Mean dist.= ' num2str(meandist) 'mm',...
                                ', Volume of active voxels=' num2str(totalVolume2) 'mm3', ...
                                ', Reg. Value=' num2str(par.reg), ', C_dist=' num2str(c_dist) 'mm',...
                                ', Peak location(in grid)= [' num2str(hspot) ']mm', ...
                                ', Centroid location=[' num2str(cntrd_locs1*ft_scalingfactor(headmodel.unit,'mm')) ']mm',...
                                ', Loc. Error=' num2str(difff) 'mm' ,...
                                ', Peak Value=' num2str(hval)]);
        annotation('textbox',[.45 .06 .52 .18], 'LineWidth', 0.0001,'FontSize', 13,...
                    'String',messagee, 'BackgroundColor', [1 1 1], 'Interpreter', 'none')
        try
            %save2word(result_figfile); close()
            saveas(gcf, [data_path1 dfname '-source.jpeg'], 'jpg'), close
        catch err
            disp(err),  close()
            disp('cant print this')
        end

        if isequal(par.resultplotsave, 'yes')
            saveas(gcf, [out_path 'Plots//' 'FT_BF_' mfname '-' stimcat '_chan-' par.meg{1,meg}], 'tiff')
            %print([out_path 'FT_' par.site '_multimodal_results//' mfname '-' stimcat '-' par.meg{1,meg}], '-dpng', '-r0');
            close(gcf)
        else
        end
    end
    clear  seg_mri source_int sourcepreM1 sourcepstM1 STC idx C sumd D
    close all
            
    end
end