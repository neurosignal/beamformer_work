function homedir=EN_add_toolboxes(add_ft, add_spm)

if ispc
    if isequal(add_ft, 'yes')
        addpath('C:\Users\fijaiami\Documents\GitHub\fieldtrip')
        ft_defaults
    end
    if isequal(add_spm, 'yes')
        addpath(genpath('C:\Users\fijaiami\Documents\spm12_Aug2018\spm12\'))
        %%addpath(genpath('C:\Users\fijaiami//Documents//MATLAB//spm12-sssbf//spm12'))
        spm('defaults', 'eeg');
    end
    homedir='C:\Users\fijaiami\';
elseif isunix
    if isequal(add_ft, 'yes')
        addpath('/home/amit/git/ChildBrain/BeamComp/MATLAB/fieldtrip-18092018')
        ft_defaults
    end
    if isequal(add_spm, 'yes')
        addpath(genpath('/home/amit/git/ChildBrain/BeamComp/MATLAB//spm12-sssbf/spm12/'))
        %addpath('/net/bonsai/home/amit/Documents/MATLAB/spm12-sssbf/spm12/')
        spm('defaults', 'eeg');
    end
    homedir='/home/amit/';
else
end