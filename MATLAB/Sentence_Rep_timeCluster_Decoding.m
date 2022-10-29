%% Init
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
Task.Conds(1).Field(1).Name='StartPart';
Task.Conds(1).Field(1).Epoch='Start';
Task.Conds(1).Field(1).Time=[0 500];
Task.Conds(1).Field(2).Name='AuditorywDelay';
Task.Conds(1).Field(2).Epoch='Auditory';
Task.Conds(1).Field(2).Time=[-500 1500];
Task.Conds(1).Field(3).Name='DelaywGo';
Task.Conds(1).Field(3).Epoch='Go';
Task.Conds(1).Field(3).Time=[-500 1500];
Task.Conds(1).Field(4).Name='AuditorytoResponse';
Task.Conds(1).Field(4).Epoch='Auditory';
Task.Conds(1).Field(4).Time=[-1000 3000];
Task.Conds(1).Field(5).Name='Response';
Task.Conds(1).Field(5).Epoch='ResponseStart';
Task.Conds(1).Field(5).Time=[-1000 1000];

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
Task.Conds(3).Field(1).Name='StartPart';
Task.Conds(3).Field(1).Epoch='Start';
Task.Conds(3).Field(1).Time=[0 500];
Task.Conds(3).Field(2).Name='AuditorywDelay';
Task.Conds(3).Field(2).Epoch='Auditory';
Task.Conds(3).Field(2).Time=[-500 1500];
Task.Conds(3).Field(3).Name='DelaywGo';
Task.Conds(3).Field(3).Epoch='Go';
Task.Conds(3).Field(3).Time=[-500 1500];
Task.Conds(3).Field(4).Name='AuditorytoResponse';
Task.Conds(3).Field(4).Epoch='Auditory';
Task.Conds(3).Field(4).Time=[-1000 3000];

Task.Conds(4).Name='JLsentences';
Task.Conds(4).Field(1).Name='AuditorywDelay';
Task.Conds(4).Field(1).Epoch='Auditory';
Task.Conds(4).Field(1).Time=[-500 5000];
Task.Conds(4).Field(2).Name='DelaywGo';
Task.Conds(4).Field(2).Epoch='Go';
Task.Conds(4).Field(2).Time=[-500 4000];

Task.Conds(5).Name='LMwords';
Task.Conds(5).Field(1).Name='StartPart';
Task.Conds(5).Field(1).Epoch='Start';
Task.Conds(5).Field(1).Time=[0 500];
Task.Conds(5).Field(2).Name='AuditorywDelay';
Task.Conds(5).Field(2).Epoch='Auditory';
Task.Conds(5).Field(2).Time=[-500 1500];
Task.Conds(5).Field(3).Name='DelaywGo';
Task.Conds(5).Field(3).Epoch='Go';
Task.Conds(5).Field(3).Time=[-500 1500];
Task.Conds(5).Field(4).Name='AuditorytoResponse';
Task.Conds(5).Field(4).Epoch='Auditory';
Task.Conds(5).Field(4).Time=[-1000 3000];
Task.Conds(5).Field(5).Name='Response';
Task.Conds(5).Field(5).Epoch='ResponseStart';
Task.Conds(5).Field(5).Time=[-1000 1000];

%% Load and Epoch Subject Data
if ~exist('Subject','var')
    Subject = popTaskSubjectData(Task);
    Subject([25:28, 30:32]) = [];
end

chanBinAll = loadTaskSubjectData(Task,Subject,DUKEDIR);

% Remove Unprocessed Data
Subject_old = Subject;
chanBinAll_old = chanBinAll;
Subject = Subject_old(~cellfun('isempty',chanBinAll_old));
chanBinAll = chanBinAll_old(~cellfun('isempty',chanBinAll_old));
if isequal(Subject_old, Subject)
    clear Subject_old chanBinAll_old
end

global sigMatChansName
[sigChans, sigMatA, allSigZ, sigMatChansName, sigMatChansLoc] = getChanSigs(Task, Subject, chanBinAll);

%save('data.mat','sigChans','sigMatA','allSigZ','sigMatChansName','sigMatChansLoc','Task','Subject')
%save('chanbinall.mat','chanBinAll','Subject','Task','-v7.3')

%load('../data/pydata.mat')
sigMatStartTime=struct;
sigMatLength=struct;
for iC=1:size(chanBinAll{1},2)
    cond = Task.Conds(iC).Name;
    for iF=1:size(chanBinAll{1}{iC},2)
        field = Task.Conds(iC).Field(iF).Name;
        for iChan=1:size(sigChans.(cond).(field),1)
            ii=find(sigMatA.(cond).(field)(sigChans.(cond).(field)(iChan),:)==1);
            sigMatStartTime.(cond).(field)(iChan)=ii(1);
            sigMatLength.(cond).(field)(iChan)=length(ii);
        end
    end
end



AUD1=intersect(sigChans{1}{1},sigChans{5}{1}); % LS LM
AUD2=intersect(AUD1,sigChans{3}{1}); % LS/LM JL
PROD1=intersect(sigChans{1}{2},sigChans{5}{2}); % LS LM
SM=intersect(AUD1,PROD1); % AUD PROD
AUD=setdiff(AUD2,SM);
PROD=setdiff(PROD1,SM);


%%


% these are significant channels that start after aud onset
[SMLS idxSM]=intersect(sigChans{1}{1},SM);
[AUDLS idxAUD]=intersect(sigChans{1}{1},AUD);


iiSM=find(sigMatStartTime{1}{1}(idxSM)>=50);
iiAUD=find(sigMatStartTime{1}{1}(idxAUD)>=50);

SM2=intersect(SM,sigChans{1}{1}(idxSM(iiSM)));
AUD2=intersect(AUD,sigChans{1}{1}(idxAUD(iiAUD)));

numFolds=10; %; 5%?

for iC=1:1 %size(chanBinAll{1},2)
    for iF=1:size(chanBinAll{iSN}{iC},2)
        elecCounter=0;
        for iSN=1:size(chanBinAll,2)
            [condIdx noiseIdx noResponseIdx]=SentenceRepConds(Subject(iSN).Trials);
            load([DUKEDIR '\Stats\timePerm\' Subject(iSN).Name '_' Subject(iSN).Task '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '.mat']);
            
            if iC<=2
                Trials=Subject(iSN).Trials(setdiff(find(condIdx==iC),cat(2,noiseIdx,noResponseIdx)));
            else
                Trials=Subject(iSN).Trials(setdiff(find(condIdx==iC),noiseIdx));
            end
            labels=[];
            for iTrials=1:length(Trials);
                labels(iTrials)=Trials(iTrials).StartCode;
            end
            
            
            labelsCount=[];
            labelsIdent={};
            labelsIdentShuff=[];
            labelsUnique=unique(labels);
            for iLabels=1:length(labelsUnique)
                labelsCount(iLabels)=length(find(labels==iLabels));
                labelsIdent{iLabels}=find(labels==iLabels);
            end
            labelsMin=min(labelsCount);
            for iLabels=1:length(labelsUnique)
                sIdx=shuffle(1:length(labelsIdent{iLabels}));
                sIdx=sIdx(1:labelsMin);
                labelsIdentShuff=cat(2,labelsIdentShuff,labelsIdent{iLabels}(sIdx));
            end
            labelsIdentShuff=sort(labelsIdentShuff);
            
            labels=labels(labelsIdentShuff);
            ieegCARHGZs=ieegCARHGZ(:,labelsIdentShuff,:);
            
            
            elecIdentGlobal=(1:size(ieegCARHGZs,1))+elecCounter;
            elecIdentLocal=1:size(ieegCARHGZs,1);
            
            SMsGlobal=intersect(SM,elecIdentGlobal);
            AUDsGlobal=intersect(AUD,elecIdentGlobal);
            PRODsGlobal=intersect(PROD,elecIdentGlobal);
            SMsLocal=SMsGlobal-elecCounter;
            AUDsLocal=AUDsGlobal-elecCounter;
            PRODsLocal=PRODsGlobal-elecCounter;
            
            
            SMConf=[];
            if ~isempty(SMsGlobal)
                for iChan=1:length(SMsGlobal)
                    tmp=sq(ieegCARHGZs(SMsLocal(iChan),:,51:size(ieegCARHGZs,3)-50));
                    sig2analyzeAllFeature=tmp(:,find(sigMatA{iC}{iF}(SMsGlobal(iChan),:)==1));
                    numDim=size(sig2analyzeAllFeature,2)-20;
                    
                    accAll = 0;
                    ytestAll = [];
                    ypredAll = [];
                    if(numFolds>0)
                        cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
                    else
                        cvp = cvpartition(labels,'LeaveOut');
                    end
                    tic
                    for nCv = 1:cvp.NumTestSets
                        nCv
                        train = cvp.training(nCv);
                        test = cvp.test(nCv);
                        ieegTrain = sig2analyzeAllFeature(train,:);
                        ieegTest = sig2analyzeAllFeature(test,:);
                        %         matTrain = size(ieegTrain);
                        %         gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
                        %         matTest = size(ieegTest);
                        %         gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);
                        
                        pTrain = labels(train);
                        pTest = labels(test);
                        gTrain=ieegTrain;
                        gTest=ieegTest;
                        [lossVect,aucVect] = scoreSelect(gTrain,pTrain,numDim,numFolds); % Hyper parameter tuning
                        [~,optimDim] = min(mean(lossVect,1)); % Selecting the optimal principal components
                        %        mean(squeeze(aucVect(:,nDim,:)),1)
                        [lossMod,Cmat,yhat,aucVect] = pcaDecode(gTrain,gTest,pTrain,...
                            pTest,optimDim);
                        ytestAll = cat(2,ytestAll,pTest);
                        ypredAll = cat(1,ypredAll,yhat);
                        accAll = accAll + 1 - lossMod;
                    end
                    CmatAll = confusionmat(ytestAll,ypredAll);
                    acc = trace(CmatAll)/sum(sum(CmatAll));
                    SMConf(iChan).CmatNorm = CmatAll./sum(CmatAll,2);
                    
                    
                end
            end
            
            AUDConf=[];
            if ~isempty(AUDsGlobal)
                for iChan=1:length(AUDsGlobal)
                    tmp=sq(ieegCARHGZs(AUDsLocal(iChan),:,51:size(ieegCARHGZs,3)-50));
                    sig2analyzeAllFeature=tmp(:,find(sigMatA{iC}{iF}(AUDsGlobal(iChan),:)==1));
                    numDim=size(sig2analyzeAllFeature,2)-5;
                    
                    accAll = 0;
                    ytestAll = [];
                    ypredAll = [];
                    if(numFolds>0)
                        cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
                    else
                        cvp = cvpartition(labels,'LeaveOut');
                    end
                    tic
                    for nCv = 1:cvp.NumTestSets
                        nCv
                        train = cvp.training(nCv);
                        test = cvp.test(nCv);
                        ieegTrain = sig2analyzeAllFeature(train,:);
                        ieegTest = sig2analyzeAllFeature(test,:);
                        %         matTrain = size(ieegTrain);
                        %         gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
                        %         matTest = size(ieegTest);
                        %         gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);
                        
                        pTrain = labels(train);
                        pTest = labels(test);
                        gTrain=ieegTrain;
                        gTest=ieegTest;
                        [lossVect,aucVect] = scoreSelect(gTrain,pTrain,numDim,numFolds); % Hyper parameter tuning
                        [~,optimDim] = min(mean(lossVect,1)); % Selecting the optimal principal components
                        %        mean(squeeze(aucVect(:,nDim,:)),1)
                        [lossMod,Cmat,yhat,aucVect] = pcaDecode(gTrain,gTest,pTrain,...
                            pTest,optimDim);
                        ytestAll = cat(2,ytestAll,pTest);
                        ypredAll = cat(1,ypredAll,yhat);
                        accAll = accAll + 1 - lossMod;
                    end
                    CmatAll = confusionmat(ytestAll,ypredAll);
                    acc = trace(CmatAll)/sum(sum(CmatAll));
                    AUDConf(iChan).CmatNorm = CmatAll./sum(CmatAll,2);
                end
            end
            
            PRODConf=[];
            if ~isempty(PRODsGlobal)
                for iChan=1:length(PRODsGlobal)
                    tmp=sq(ieegCARHGZs(PRODsLocal(iChan),:,51:size(ieegCARHGZs,3)-50));
                    sig2analyzeAllFeature=tmp(:,find(sigMatA{iC}{iF}(PRODsGlobal(iChan),:)==1));
                    numDim=size(sig2analyzeAllFeature,2)-5;
                    
                    accAll = 0;
                    ytestAll = [];
                    ypredAll = [];
                    if(numFolds>0)
                        cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
                    else
                        cvp = cvpartition(labels,'LeaveOut');
                    end
                    tic
                    for nCv = 1:cvp.NumTestSets
                        nCv
                        train = cvp.training(nCv);
                        test = cvp.test(nCv);
                        ieegTrain = sig2analyzeAllFeature(train,:);
                        ieegTest = sig2analyzeAllFeature(test,:);
                        %         matTrain = size(ieegTrain);
                        %         gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
                        %         matTest = size(ieegTest);
                        %         gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);
                        
                        pTrain = labels(train);
                        pTest = labels(test);
                        gTrain=ieegTrain;
                        gTest=ieegTest;
                        [lossVect,aucVect] = scoreSelect(gTrain,pTrain,numDim,numFolds); % Hyper parameter tuning
                        [~,optimDim] = min(mean(lossVect,1)); % Selecting the optimal principal components
                        %        mean(squeeze(aucVect(:,nDim,:)),1)
                        [lossMod,Cmat,yhat,aucVect] = pcaDecode(gTrain,gTest,pTrain,...
                            pTest,optimDim);
                        ytestAll = cat(2,ytestAll,pTest);
                        ypredAll = cat(1,ypredAll,yhat);
                        accAll = accAll + 1 - lossMod;
                    end
                    CmatAll = confusionmat(ytestAll,ypredAll);
                    acc = trace(CmatAll)/sum(sum(CmatAll));
                    PRODConf(iChan).CmatNorm = CmatAll./sum(CmatAll,2);
                end
            end
            save([DUKEDIR '\Stats\timePerm\Decoding\' Subject(SN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '_' num2str(numFolds) 'Folds.mat'],'SMConf','AUDConf','PRODConf');
            elecCounter=elecCounter+size(ieegCARHGZ,1);
        end
    end
end
            