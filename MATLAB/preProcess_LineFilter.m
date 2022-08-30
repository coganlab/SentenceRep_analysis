function preProcess_LineFilter(Task, Subjects)
% Task is a structure with .Name = Task Name


global RECONDIR
global TASK_DIR
global experiment
global DUKEDIR
global BOX_DIR
if isempty(BOX_DIR)
    BOX_DIR='C:\Users\ae166\Box';
end
RECONDIR=[BOX_DIR '\ECoG_Recon'];
DUKEDIR = [BOX_DIR '\CoganLab\D_Data\SentenceRep'];

%addpath(genpath([BOX_DIR '\CoganLab\Scripts\']));

TASK_DIR=DUKEDIR;
%TASK_DIR=([BOX_DIR '\CoganLab\D_Data\' Task.Name]);
%DUKEDIR=TASK_DIR;


% Populate Subject file
if ~exist('Subjects','var')
    Subjects = popTaskSubjectData(Task);
end


for iSN=1:length(Subjects)
    Subject = Subjects(iSN);
    load([TASK_DIR '/' Subject.Name '/mat/experiment.mat']);
    load([TASK_DIR '/' Subject.Name '/' Subject.Date '/mat/Trials.mat'])
    
    % linefilter if not already done
    for iR=1:length(Subject.Rec)
        if Subject.Rec(iR).lineNoiseFiltered==0
            display(['filtering ' Subject.Name ' task ' Subject.Rec(iR).fileNamePrefix])
            ntools_procCleanIEEG([TASK_DIR '/' Subject.Name '/' ...
                Subject.Date '/00' num2str(iR) ...
                '/' Subject.Rec(iR).fileNamePrefix]);
        end
    end
end

