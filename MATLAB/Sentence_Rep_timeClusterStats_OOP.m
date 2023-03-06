clear; clc;
global BOX_DIR
global RECONDIR
global DUKEDIR
BOX_DIR=[userpath '\..\..\Box'];
%userpath is usually %USERPROFILE%/Documents/MATLAB, so move up two
%directories to get to %USERPROFILE%/Box
RECONDIR=[BOX_DIR filesep 'ECoG_Recon'];
DUKEDIR = fullfile(BOX_DIR, 'CoganLab', 'D_Data', 'SentenceRep');
path = fullfile(userpath, 'MATLAB-env' );
addpath(genpath(path));
path = fullfile('..','IEEG_Pipelines','MATLAB');
addpath(genpath(path));

%% Task
Task=[];
Task.Name='SentenceRep';

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-0.5 0];

% old nyu baseline (deprecated)
% Task.Base.Name='AuditoryPre';
% Task.Base.Epoch='Auditory';
% Task.Base.Time=[-500 0];

% 5 conds
Task.Conds(1).Name='LSwords';
% Task.Conds(1).Field(1).Name='StartPart';
% Task.Conds(1).Field(1).Epoch='Start';
% Task.Conds(1).Field(1).Time=[0 500];
Task.Conds(1).Field(1).Name='AuditorywDelay';
Task.Conds(1).Field(1).Epoch='Auditory';
Task.Conds(1).Field(1).Time=[-0.5 1.5];
Task.Conds(1).Field(2).Name='DelaywGo';
Task.Conds(1).Field(2).Epoch='Go';
Task.Conds(1).Field(2).Time=[-0.5 1.5];
% Task.Conds(1).Field(4).Name='AuditorytoResponse';
% Task.Conds(1).Field(4).Epoch='Auditory';
% Task.Conds(1).Field(4).Time=[-1000 3000];
Task.Conds(1).Field(3).Name='Response';
Task.Conds(1).Field(3).Epoch='ResponseStart';
Task.Conds(1).Field(3).Time=[-1 1];

Task.Conds(2).Name='LSsentences';
Task.Conds(2).Field(1).Name='AuditorywDelay';
Task.Conds(2).Field(1).Epoch='Auditory';
Task.Conds(2).Field(1).Time=[-0.5 5];
Task.Conds(2).Field(2).Name='DelaywGo';
Task.Conds(2).Field(2).Epoch='Go';
Task.Conds(2).Field(2).Time=[-0.5 4];
Task.Conds(2).Field(3).Name='Response';
Task.Conds(2).Field(3).Epoch='ResponseStart';
Task.Conds(2).Field(3).Time=[-1 4];

Task.Conds(3).Name='JLwords';
Task.Conds(3).Field(1).Name='AuditorywDelay';
Task.Conds(3).Field(1).Epoch='Auditory';
Task.Conds(3).Field(1).Time=[-0.5 1.5];
Task.Conds(3).Field(2).Name='DelaywGo';
Task.Conds(3).Field(2).Epoch='Go';
Task.Conds(3).Field(2).Time=[-0.5 1.5];

Task.Conds(4).Name='JLsentences';
Task.Conds(4).Field(1).Name='AuditorywDelay';
Task.Conds(4).Field(1).Epoch='Auditory';
Task.Conds(4).Field(1).Time=[-0.5 5];
Task.Conds(4).Field(2).Name='DelaywGo';
Task.Conds(4).Field(2).Epoch='Go';
Task.Conds(4).Field(2).Time=[-0.5 4];

Task.Conds(5).Name='LMwords';
Task.Conds(5).Field(1).Name='AuditorywDelay';
Task.Conds(5).Field(1).Epoch='Auditory';
Task.Conds(5).Field(1).Time=[-0.5 1.5];
Task.Conds(5).Field(2).Name='DelaywGo';
Task.Conds(5).Field(2).Epoch='Go';
Task.Conds(5).Field(2).Time=[-0.5 1.5];

if ~exist('Subject','var')
    Subject = popTaskSubjectData(Task);
end

%%
fDown = 100; %Downsampled Sampling Frequency
timeExtract = [-1.5 2];

% HGZ_data_start = extractHGDataWithROI(Subject, Epoch='Start', fdown=100, ...
%     Time=[0,0.5], baseTimeRange=baseTimeRange, baseName=baseName);
% HGZ_data_aud = extractHGDataWithROI(Subject, Epoch='Auditory', fdown=100, ...
%     Time=[-0.5,1.5], baseTimeRange=baseTimeRange, baseName=baseName);
% HGZ_data_go = extractHGDataWithROI(Subject, Epoch='Go', fdown=100, ...
%     Time=[-0.5,1.5], baseTimeRange=baseTimeRange, baseName=baseName);
% HGZ_data_resp = extractHGDataWithROI(Subject, Epoch='ResponseStart', fdown=100, ...
%     Time=[-1,1], baseTimeRange=baseTimeRange, baseName=baseName);
% alldat = {HGZ_data_start,HGZ_data_aud,HGZ_data_go};%,HGZ_data_resp};
% conds = {'start','aud','go'};%,'resp'};
%%
SNList=1:length(Subject);
for iSN=1:length(SNList)
    SN=SNList(iSN);
    Trials=Subject(SN).Trials;
    counterN=0;
    counterNR=0;
    noiseIdx=0;
    noResponseIdx=0;
    for iTrials=1:length(Trials)
        if Trials(iTrials).Noisy==1
            noiseIdx(counterN+1)=iTrials;
            counterN=counterN+1;
        end
        if Trials(iTrials).NoResponse==1
            noResponseIdx(counterNR+1)=iTrials;
            counterNR=counterNR+1;
        end
    end
   
    condIdx=ones(length(Subject(SN).Trials),1);
    chanIdx=Subject(SN).goodChannels;
    baseEpoch=Task.Base.Epoch;
    baseTimeRange=Task.Base.Time;    
    Trials=Subject(SN).Trials(setdiff(1:length(Subject(SN).Trials),noiseIdx));

    ieegBase=trialIEEG(Trials,chanIdx,baseEpoch,timeExtract.*1000);
    ieegBase = permute(ieegBase,[2,1,3]);
    fs = Subject(SN).Experiment.processing.ieeg.sample_rate;   
    ieegBaseStruct = ieegStructClass(ieegBase, fs, timeExtract, [1 fs/2], baseEpoch);

    clear ieegBase;
    ieegBaseCAR=extractCar(ieegBaseStruct);
    clear ieegBaseStruct;
    ieegBaseHG = extractHiGamma(ieegBaseCAR,fDown,baseTimeRange);
   
    for iC=1:length(Task.Conds)
        for iF=1:length(Task.Conds(iC).Field)
          %  if iC<=2
                Trials=Subject(SN).Trials(setdiff(find(condIdx==iC),cat(2,noiseIdx,noResponseIdx)));
         %   else
         %       Trials=Subject(SN).Trials(setdiff(find(condIdx==iC),noiseIdx));
         %   end
            
            Epoch=Task.Conds(iC).Field(iF).Epoch;
            fieldTimeRange=Task.Conds(iC).Field(iF).Time;          
                        
            ieegField=trialIEEG(Trials,chanIdx,Epoch,timeExtract.*1000);
            ieegField = permute(ieegField,[2,1,3]);
            ieegFieldStruct = ieegStructClass(ieegField, fs, timeExtract, [1 fs/2], Epoch);
            
            clear ieegField
            % Common average referencing
            ieegFieldCAR = extractCar(ieegFieldStruct);
            clear ieegFieldStruct;
            % High gamma extraction
            ieegFieldHG = extractHiGamma(ieegFieldCAR,fDown,fieldTimeRange);    
            % Time Series Permutation cluster
            chanSig = extractTimePermCluster(ieegFieldHG,ieegBaseHG);         
            channelNames = {Subject(SN).ChannelInfo(chanIdx).Name};
            
            if ~exist([DUKEDIR '\Stats\timePerm\'])
                mkdir([DUKEDIR '\Stats\timePerm\'])
            end
            save(fullfile(DUKEDIR, 'Stats', 'timePerm', [Subject(SN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat']),'chanSig','channelNames','ieegFieldHG','ieegBaseHG');
        end
    end
end