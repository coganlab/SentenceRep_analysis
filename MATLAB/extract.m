clear; clc;
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
path = fullfile('..','IEEG_Pipelines','MATLAB');
addpath(genpath(path));

%% Task
Task=[];
Task.Name='SentenceRep';

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-1000 -500];

% old nyu baseline (deprecated)
% Task.Base.Name='AuditoryPre';
% Task.Base.Epoch='Auditory';
% Task.Base.Time=[-500 0];

% 5 conds
Task.Conds(1).Name='LSwords';
Task.Conds(1).Field(1).Name='AuditorywDelay';
Task.Conds(1).Field(1).Epoch='Auditory';
Task.Conds(1).Field(1).Time=[-500 1500];
Task.Conds(1).Field(2).Name='DelaywGo';
Task.Conds(1).Field(2).Epoch='Go';
Task.Conds(1).Field(2).Time=[-500 1500];
Task.Conds(1).Field(3).Name='Response';
Task.Conds(1).Field(3).Epoch='ResponseStart';
Task.Conds(1).Field(3).Time=[-1000 1000];

Task.Conds(2).Name='LSsentences';
Task.Conds(2).Field(1).Name='AuditorywDelay';
Task.Conds(2).Field(1).Epoch='Auditory';
Task.Conds(2).Field(1).Time=[-500 5000];
Task.Conds(2).Field(2).Name='DelaywGo';
Task.Conds(2).Field(2).Epoch='Go';
Task.Conds(2).Field(2).Time=[-500 4000];
Task.Conds(2).Field(3).Name='Response';
Task.Conds(2).Field(3).Epoch='ResponseStart';
Task.Conds(2).Field(3).Time=[-1000 4000];

Task.Conds(3).Name='JLwords';
Task.Conds(3).Field(1).Name='AuditorywDelay';
Task.Conds(3).Field(1).Epoch='Auditory';
Task.Conds(3).Field(1).Time=[-500 1500];
Task.Conds(3).Field(2).Name='DelaywGo';
Task.Conds(3).Field(2).Epoch='Go';
Task.Conds(3).Field(2).Time=[-500 1500];

Task.Conds(4).Name='JLsentences';
Task.Conds(4).Field(1).Name='AuditorywDelay';
Task.Conds(4).Field(1).Epoch='Auditory';
Task.Conds(4).Field(1).Time=[-500 5000];
Task.Conds(4).Field(2).Name='DelaywGo';
Task.Conds(4).Field(2).Epoch='Go';
Task.Conds(4).Field(2).Time=[-500 4000];

Task.Conds(5).Name='LMwords';
Task.Conds(5).Field(1).Name='AuditorywDelay';
Task.Conds(5).Field(1).Epoch='Auditory';
Task.Conds(5).Field(1).Time=[-500 1500];
Task.Conds(5).Field(2).Name='DelaywGo';
Task.Conds(5).Field(2).Epoch='Go';
Task.Conds(5).Field(2).Time=[-500 1500];

if ~exist('Subject','var')
    Subject = popTaskSubjectData(Task);
    Subject([25:28, 30:32]) = [];
end

%%
baseTimeRange = Task.Base.Time/1000;
baseName = Task.Base.Epoch;

HGZ_data_start = extractHGDataWithROI(Subject, Epoch='Start', fdown=100, ...
    Time=[0,0.5], baseTimeRange=baseTimeRange, baseName=baseName);
HGZ_data_aud = extractHGDataWithROI(Subject, Epoch='Auditory', fdown=100, ...
    Time=[-0.5,1.5], baseTimeRange=baseTimeRange, baseName=baseName);
HGZ_data_go = extractHGDataWithROI(Subject, Epoch='Go', fdown=100, ...
    Time=[-0.5,1.5], baseTimeRange=baseTimeRange, baseName=baseName);
HGZ_data_resp = extractHGDataWithROI(Subject, Epoch='ResponseStart', fdown=100, ...
    Time=[-1,1], baseTimeRange=baseTimeRange, baseName=baseName);
alldat = {HGZ_data_start,HGZ_data_aud,HGZ_data_go,HGZ_data_resp};
conds = {'start','aud','go','resp'};
%%
subj = Subject;
subj([15,23,25,27])=[];

trialdata = extractTrialInfo(Subject);
trialdata([15,23,25,27])=[];
% cellfun(@sentRepSort,trialdata,'UniformOutput',false)
data = struct();
data.ls = struct();
trialInfo = {};
% data.lm = struct();
% data.jl = struct();
condNum = [1,1]; % LS
cond = 'ls';
% condNum = [1,2] % LM
% condNum = [2,3] % JL
for iC = 1:4
tempdata = [alldat{iC}.ieegHGNorm];
celldata = {tempdata.data};
celldata([15,23,25,27]) = [];
% find the minimum trials per subject for the condition
minIdx = 100;
for iSN = length(trialdata):-1:1
    condIdx = sentRepSort(trialdata{iSN});
    minIdx = min(minIdx, sum(condIdx(:,1)==condNum(1) & condIdx(:,3)==condNum(2)));
end
lsIdx = find(condIdx(:,1)==1 & condIdx(:,3)==1);
data.ls.(conds{iC}) = celldata{1}(:,lsIdx(1:minIdx),:);
trialInfo{iSN} = trialdata{iSN}(lsIdx(1:minIdx));
for iSN = 2:length(trialdata)
    condIdx = sentRepSort(trialdata{iSN});
    lsIdx = find(condIdx(:,1)==condNum(1) & condIdx(:,3)==condNum(2));
    data.ls.(conds{iC}) = cat(1,data.ls.(conds{iC}), celldata{iSN}(:,lsIdx(1:minIdx),:));
    trialInfo{iSN} = trialdata{iSN}(lsIdx(1:minIdx));
    % data.ls.word(iSN) = [data.ls.trialInfo,[trialdata{iSN}(lsIdx(1:minIdx))].sound];
end
end

%% 
listenSpeak = data.ls;
dat = HGZ_data_start;
dat([15,23,25,27])=[];
channelNames = [dat.channelName];
save('../data/pydata_3d.mat','listenSpeak','trialInfo','subj','channelNames')