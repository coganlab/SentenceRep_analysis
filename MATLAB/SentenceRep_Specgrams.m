%clear; clc
global BOX_DIR
global RECONDIR
global DUKEDIR
BOX_DIR=[userpath '\..\..\Box'];
%userpath is usually %USERPROFILE%/Documents/MATLAB, so move up two
%directories to get to %USERPROFILE%/Box
RECONDIR=[BOX_DIR '\ECoG_Recon'];
DUKEDIR = [BOX_DIR '\CoganLab\D_Data\SentenceRep'];
path = fullfile(userpath, 'MATLAB-env' );
addpath(genpath(path));

Task=[];
Task.Name='SentenceRep';

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-500, 0];

% old nyu baseline (deprecated)
% Task.Base.Name='AuditoryPre';
% Task.Base.Epoch='Auditory';
% Task.Base.Time=[-500 0];

% 5 conds
Task.Conds(1).Name='LSwords';
Task.Conds(1).Field(1).Name='AuditorywDelay';
Task.Conds(1).Field(1).Epoch='Auditory';
Task.Conds(1).Field(1).Time=[0 1000];
Task.Conds(1).Field(2).Name='DelaywGo';
Task.Conds(1).Field(2).Epoch='Go';
Task.Conds(1).Field(2).Time=[0 1000];
% Task.Conds(1).Field(3).Name='Response';
% Task.Conds(1).Field(3).Epoch='ResponseStart';
% Task.Conds(1).Field(3).Time=[-1000 1000];

Task.Conds(2).Name='LSsentences';
Task.Conds(2).Field(1).Name='AuditorywDelay';
Task.Conds(2).Field(1).Epoch='Auditory';
Task.Conds(2).Field(1).Time=[-500 5000];
Task.Conds(2).Field(2).Name='DelaywGo';
Task.Conds(2).Field(2).Epoch='Go';
Task.Conds(2).Field(2).Time=[-500 4000];
% Task.Conds(2).Field(3).Name='Response';
% Task.Conds(2).Field(3).Epoch='ResponseStart';
% Task.Conds(2).Field(3).Time=[-1000 4000];

Task.Conds(3).Name='LMwords';
Task.Conds(3).Field(1).Name='AuditorywDelay';
Task.Conds(3).Field(1).Epoch='Auditory';
Task.Conds(3).Field(1).Time=Task.Conds(1).Field(1).Time;
Task.Conds(3).Field(2).Name='DelaywGo';
Task.Conds(3).Field(2).Epoch='Go';
Task.Conds(3).Field(2).Time=Task.Conds(1).Field(2).Time;


Task.Conds(4).Name='JLsentences';
Task.Conds(4).Field(1).Name='AuditorywDelay';
Task.Conds(4).Field(1).Epoch='Auditory';
Task.Conds(4).Field(1).Time=Task.Conds(2).Field(1).Time;
Task.Conds(4).Field(2).Name='DelaywGo';
Task.Conds(4).Field(2).Epoch='Go';
Task.Conds(4).Field(2).Time=Task.Conds(2).Field(2).Time;

Task.Conds(5).Name='JLwords';
Task.Conds(5).Field(1).Name='AuditorywDelay';
Task.Conds(5).Field(1).Epoch='Auditory';
Task.Conds(5).Field(1).Time=Task.Conds(1).Field(1).Time;
Task.Conds(5).Field(2).Name='DelaywGo';
Task.Conds(5).Field(2).Epoch='Go';
Task.Conds(5).Field(2).Time=Task.Conds(1).Field(2).Time;
Task.Fig = struct();
Task.Fig.Field = struct();
fNum = 0;

for iC = 1:length(Task.Conds)
    for iF=1:length(Task.Conds(iC).Field)
        Task.Fig.Field(iF + fNum).Name = [Task.Conds(iC).Name, '_', Task.Conds(iC).Field(iF).Name];
        Task.Fig.Field(iF + fNum).Epoch = Task.Conds(iC).Field(iF).Epoch;
        Task.Fig.Field(iF + fNum).Time = Task.Conds(iC).Field(iF).Time;
    end
    fNum = fNum + iF;
end

Task.Fig.Baseline.Name=Task.Base.Name;
Task.Fig.Baseline.Epoch=Task.Base.Epoch;
Task.Fig.Baseline.Time=Task.Base.Time;

if ~exist('Subjects','var')
    Subjects = popTaskSubjectData(Task);
%     Subject([25:28, 30:32]) = [];
end
Subject = getSubjects(Subjects, 'D73');

preProcess_Specgrams(Task,Subject,'eps')


