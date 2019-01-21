%% Usage   : LCMV beamformer pipeline for neuromag dataset using FieldTrip
%% Scripted: Amit Jaiswal @ MEGIN, Helsinki, Finland 
%% Support : support@megin.fi (CC to amit.jaiswal@megin.fi)
%% Created : 15 Dec. 2018, Version: 1.0.2
%% File No : NM/FT0922C1512
%% ***********************************************************************
function [headmodel, leadfield] = NM_ft_prepare_forwardmodel(par, mrifname, segmrifname, fname, data_summary, varargin)
%     leadfield   = ft_getopt(varargin, 'leadfield');
%     headmodel   = ft_getopt(varargin, 'headmodel');
    
    %% Prepare headmodel and compute leadfield
    if exist(segmrifname, 'file')==2 && ~isequal(par.mri_seg, 'yes')
        load(segmrifname); %% load the saves segmented volume
    else
        mri = ft_read_mri(mrifname); % read mri
        if isequal(par.mri_realign, 'yes') && isequal(par.align_interactive, 'yes') % Interactive 
                cfg             =[];
                cfg.method      ='interactive';
                cfg.coordsys    = 'neuromag';
                cfg.parameter   = 'anatomy';
                cfg.viewresult  =  'yes' ;
                [mri] = ft_volumerealign(cfg, mri);
        elseif isequal(par.mri_realign, 'yes') &&isequal(par.align_headshape, 'yes') % Using ICP
                cfg                     = [];
                cfg.method              = 'headshape';
                cfg.headshape.headshape = ft_read_headshape(fname,'unit',mri.unit);
                cfg.headshape.icp       = 'yes';
                cfg.coordsys            = 'neuromag';
                cfg.parameter           = 'anatomy';
                cfg.viewresult          = 'yes';
                [mri] = ft_volumerealign(cfg, mri);
        end

        cfg          = [];  %% Realined MRI segmentation
        cfg.output   = 'brain';
        cfg.spmversion = 'spm12';
        segmri = ft_volumesegment(cfg, mri);
        segmri.transform = mri.transform;
        segmri.anatomy   = mri.anatomy;

        [segdir, segf, ~]=fileparts(mrifname);
        save([segdir '//' segf '-brain.mat'], 'segmri');    
    end

    if isequal(par.more_plots, 'yes') % Plot segmented MRI volume
        fprintf('\n\nPlot segmented MRI volume >>>>>>>>>>>>>>>>>>>>>>>\n\n')
        cfg              = [];
        cfg.funparameter = 'brain';
        cfg.location     = [0,0,0];
        ft_sourceplot(cfg, segmri);
        suptitle('Coregistered & segmented MRI orthogonal plots')
        set(gcf, 'Position', get(0, 'Screensize'));
    end

    %% Compute the subject's headmodel/volume conductor model
        if ~exist('headmodel', 'var')
            if ~isequal(par.mri_seg, 'yes'), load(segmrifname); end
            cfg                = [];
            cfg.method         = 'singleshell';
            % cfg.tissue         = 'brain';
            fprintf('\nPreparing %s headmodel > > > > >\n\n', cfg.method)
            headmodel          = ft_prepare_headmodel(cfg, segmri);
            if isequal(par.scale_data_fromT2fT, 'yes') && isequal(data_summary.grad.unit, 'mm')
               headmodel = ft_convert_units(headmodel, 'mm'); 
            end
        end
        if isequal(par.more_plots, 'yes')
            pause(2)
            fprintf('\n\nPlotting Headmodel >>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
            figure, ft_plot_vol(headmodel, 'facecolor', 'brain'), rotate3d % plot headmodel
            title(sprintf('Single shell headmodel with %d vertices', length(headmodel.bnd.pos)))
            %set(gcf, 'Position', get(0, 'Screensize'));
            %view(3)
            pause(1),  view(-90,1), snapnow;
            pause(1),  view(1,90), snapnow;
            pause(1),  view(0,0), snapnow;
            pause(1),  view(90,0), snapnow;
        end

    %% Cross check the alignment of MRI and isotrack points and headmodel inside sensors array
    if isequal(par.more_plots, 'yes') && isequal(par.mri_realign, 'yes')
        fprintf('\n\nCross check the alignment of MRI and isotrack points and headmodel inside sensors array >>>>>>>>\n\n')
        if ~exist('mri', 'var'); mri=ft_read_mri(mrifname); end
        ft_determine_coordsys(mri, 'interactive', 'no')
        ft_plot_headshape(ft_read_headshape(fname,'unit',segmri.unit), 'vertexsize', 20)
        ft_plot_vol(headmodel);
        ft_plot_sens(ft_convert_units(ft_read_sens(fname), segmri.unit), 'coilshape', 'circle', 'coilsize', 0.015, 'facecolor', [1 1 1])
        rotate3d
        set(gcf, 'Position', get(0, 'Screensize'));
    end

    %% Create the subject specific grid (source space)
    if ~exist('leadfield', 'var') || length(data_summary.label)~=length(leadfield.label);
        cfg                 = [];
        cfg.grad            = ft_convert_units(data_summary.grad, headmodel.unit);
        cfg.headmodel       = headmodel;
        if isequal(headmodel.unit, 'mm')
            cfg.grid.resolution = par.gridres;
        elseif isequal(headmodel.unit, 'm')
            cfg.grid.resolution = par.gridres/1000;
        end
        cfg.grid.unit       = headmodel.unit;
        cfg.inwardshift     = cfg.grid.resolution/10; 
        fprintf('\nPreparing a rectangular grid with %.3f%s resolution > > > > >\n\n',...
                cfg.grid.resolution, headmodel.unit)
        src_v               = ft_prepare_sourcemodel(cfg);
        if isequal(par.more_plots, 'yess') % plot rectangular grid
            figure, ft_plot_mesh(src_v); rotate3d 
        end
        % % Create leadfield (forward model for many dipole locations)
        cfg                 = [];
        cfg.grad            = ft_convert_units(data_summary.grad, headmodel.unit);  
        cfg.headmodel       = headmodel;% volume conduction headmodel
        cfg.grid            = src_v;    % grid positions
        cfg.channel         = data_summary.label;
        cfg.normalize       = 'yes';    % to remove depth bias
        %cfg.normalizeparam  = 0.5;
        cfg.backproject     = 'yes';
        cfg.senstype        = 'MEG';
        fprintf('\nPreparing leadfield using %d channels and grid with %.3f%s resolution > > > > >\n\n',...
                length(data_summary.label), src_v.cfg.grid.resolution, src_v.cfg.grid.unit)
        leadfield           = ft_prepare_leadfield(cfg, data_summary);

        nchan_lf = length(leadfield.label);
    else
        disp('Using leadfield computed in previous run > > > > >')
    end

    %% Cross check the alignment of isotrack points, headmodel and leadfield inside sensors array and helmet 
    if isequal(par.more_plots, 'yes')
        fprintf('\n\nCross check the alignment of isotrack points, headmodel and leadfield inside sensors array >>>>\n\n')
        headshape=ft_read_headshape(fname,'unit',segmri.unit);
        figure
        ft_plot_headshape(headshape, 'fidlabel','no', 'vertexsize', 20)
        text(headshape.fid.pos(:,1), headshape.fid.pos(:,2), headshape.fid.pos(:,3), headshape.fid.label(:), 'fontsize',10);
        text(headshape.pos(1:4,1), headshape.pos(1:4,2), headshape.pos(1:4,3), headshape.label(1:4), 'color', 'b', 'fontsize',10);
        clear headshape
        ft_plot_vol(headmodel, 'facecolor', 'skin'); alpha 0.3; camlight
        ft_plot_sens(ft_convert_units(ft_read_sens(fname, 'senstype', 'meg'), segmri.unit), 'coilshape', 'circle', 'coilsize', 0.015, 'facecolor', [1 1 1])
        ft_plot_mesh(leadfield.pos(leadfield.inside,:));
        grad=ft_convert_units(data_summary.grad, 'm'); ft_plot_topo3d(grad.chanpos, ones(306,1)), alpha 0.2; 
        title('Aligned plot : Scanning grid | head model | isotrack points | Sensor array')
        %set(gcf, 'Position', get(0, 'Screensize'));
        %rotate3d
        pause(1),  view(-90,1), snapnow; 
        pause(1),  view(1,90), snapnow;
        pause(1),  view(0,0), snapnow;
        pause(1),  view(90,0), snapnow;
    end
    clear mri src_v % to vacate memory
end