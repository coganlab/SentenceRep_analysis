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

Task=[];
Task.Name='SentenceRep';

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-1000 -500];
%
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

% Task.Conds(1).Field(4).Name='AuditorytoResponse';
% Task.Conds(1).Field(4).Epoch='Auditory';
% Task.Conds(1).Field(4).Time=[-1000 3000];

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






Subject = popTaskSubjectData(Task);


chanBinAll={};
SNList=1:length(Subject);
%SNList=1:24;
for iSN=1:length(SNList);
    SN=SNList(iSN);
    chanIdx=Subject(SN).goodChannels;
    for iC=1:length(Task.Conds)
        for iF=1:length(Task.Conds(iC).Field)
            load([DUKEDIR '\Stats\timePerm\' Subject(SN).Name '_' Subject(SN).Task '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '.mat']);
            %             load([DUKEDIR '\Stats\timePerm\' Subject(SN).Name '_' Subject(SN).Task '_' ...
            %                 Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat']);


            baseTimeRange=Task.Base.Time;
            baseTimeRange(1)=baseTimeRange(1)-500;
            baseTimeRange(2)=baseTimeRange(2)+500;
            TimeRange=Task.Conds(iC).Field(iF).Time;
            TimeRange(1)=TimeRange(1)-500;
            TimeRange(2)=TimeRange(2)+500;
            TimeLength=TimeRange(2)-TimeRange(1);

            totChanBlock=ceil(size(ieegCARHG,1)./60);
            iChan=0;
            chanSigBin=zeros(length(chanSig),1);
            tscale=linspace(TimeRange(1)+500,TimeRange(2)-500,size(ieegCARHG,3)-100);
            sigBinMat=zeros(size(ieegCARHG,1),size(ieegCARHG,3)-100);
            %     FigS=figure('Position', get(0, 'Screensize'));
            % figure
            for iChan=1:size(ieegCARHG,1) %min(60,size(ieegCARHG,1)-iChan);

                for iCl=1:length(chanSig{iChan}.actClust.Size)
                    % ii=sort(chanSig{iChan}.actClust.maxPermClust);
                    % if chanSig{iChan}.actClust.Size{iCl}>ii(990) % p<.01);
                    if chanSig{iChan}.actClust.Size{iCl}>chanSig{iChan}.actClust.perm95
                        chanSigBin(iChan)=1;
                        sigBinMat(iChan,chanSig{iChan}.actClust.Start{iCl}:chanSig{iChan}.actClust.Start{iCl}...
                            +(chanSig{iChan}.actClust.Size{iCl}-1))=ones(chanSig{iChan}.actClust.Size{iCl},1);
                    end
                end
            end
            chanBinAll{iSN}{iC}{iF}.matrix=sigBinMat;
            chanBinAll{iSN}{iC}{iF}.sigChans=chanSigBin;
        end
    end
    display(iSN)
end





%sigMat=[];
sigMatChans={};
sigMatChansLoc={};
sigMatChansName={};
sigChans={};
%sigMatAll=[];
sigMatA={};
allSigZ={};
AllSigWMIdx=[];

for iC=1:size(chanBinAll{1},2)
    for iF=1:size(chanBinAll{iSN}{iC},2)
        allSigZt=[];
        sigMatAt=[];
        sigChanst=[];
        elecCounter=0;
        for iSN=1:size(chanBinAll,2)
            sigMatChansCounter=0;
            load([DUKEDIR '\Stats\timePerm\' Subject(iSN).Name '_' Subject(iSN).Task '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '.mat']);
            %             load([DUKEDIR '\Stats\timePerm\' Subject(iSN).Name '_' Subject(iSN).Task '_' ...
            %                 Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat']);

            allSigZt=cat(1,allSigZt,sq(mean(ieegCARHGZ(:,:,51:size(ieegCARHGZ,3)-50),2)));


            sigIdxGC=find(chanBinAll{iSN}{iC}{iF}.sigChans==1);
            % get rid of WM
            [nonWMIdx nonWMIdxGC]=setdiff(Subject(iSN).goodChannels,Subject(iSN).WM);
            [WMIdx WMIdxGC]=intersect(Subject(iSN).goodChannels,Subject(iSN).WM);
            chanIdx=intersect(sigIdxGC,nonWMIdxGC);
            AllSigWMIdx=cat(1,AllSigWMIdx,WMIdxGC);
            % chanIdx=setdiff(chanIdx,Subject(iSN).WM);
            %  chanIdx2=
            %sigMat=cat(1,sigMat,chanBinAll{iSN}{iC}{iF}.matrix(chanIdx,:));
            sigMatAt=cat(1,sigMatAt,chanBinAll{iSN}{iC}{iF}.matrix(:,:));
            %  sigMatAll=cat(1,sigMatAll,sigMatA);
            sigChanst=cat(1,sigChanst,chanIdx+elecCounter);
            for iChan=1:length(Subject(iSN).goodChannels)
                sigMatChansName.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name){elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Name;
                sigMatChansLoc.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name){elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Location;
                elecCounter=elecCounter+1;
                %  sigMatChansCounter=sigMatChansCounter+1;
            end
            % elecCounter=elecCounter+size(chanBinAll{iSN}{iC}{iF}.matrix,1);
        end
        allSigZ.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)=allSigZt;
        sigMatA.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)=sigMatAt;
        sigChans.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)=sigChanst;

    end
    display(iC)
end

sigMatStartTime={};
sigMatLength={};
for iC=1:size(chanBinAll{1},2)
    for iF=1:size(chanBinAll{iSN}{iC},2)
        for iChan=1:size(sigChans.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name),1)
            ii=find(sigMatA.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)(sigChans.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)(iChan),:)==1);
            sigMatStartTime.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)(iChan)=ii(1);
            sigMatLength.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)(iChan)=length(ii);
        end
    end
end


%
% AUD1=intersect(sigChans{1}{1},sigChans{5}{1}); % LS LM
% AUD2=intersect(AUD1,sigChans{3}{1}); % LS/LM JL
% PROD1=intersect(sigChans{1}{2},sigChans{5}{2}); % LS LM
% SM=intersect(AUD1,PROD1); % AUD PROD
% AUD=setdiff(AUD2,SM);
% PROD=setdiff(PROD1,SM);
%
%
%
%
%
% % these are significant channels that start after aud onset
% [SMLS idxSM]=intersect(sigChans{1}{1},SM);
% [AUDLS idxAUD]=intersect(sigChans{1}{1},AUD);
%
%
% iiSM=find(sigMatStartTime{1}{1}(idxSM)>=50);
% iiAUD=find(sigMatStartTime{1}{1}(idxAUD)>=50);
%
% SM2=intersect(SM,sigChans{1}{1}(idxSM(iiSM)));
% AUD2=intersect(AUD,sigChans{1}{1}(idxAUD(iiAUD)));
%%
numFolds=10; %; 5%?

for iC=[1,3,5]%1:size(chanBinAll{1},2)
    for iF=1:size(chanBinAll{iSN}{iC},2)
        elecCounter=0;
        for iSN=1:size(chanBinAll,2)
            [condIdx noiseIdx noResponseIdx]=SentenceRepConds(Subject(iSN).Trials);
            load([DUKEDIR '\Stats\timePerm\' Subject(iSN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat']);

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
                labelsCount(iLabels)=length(find(labels==labelsUnique(iLabels)));
                labelsIdent{iLabels}=find(labels==labelsUnique(iLabels));
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

            chanIdx=sigChans.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name);
            chanIdxGlobal=intersect(chanIdx,elecIdentGlobal);
            chanIdxLocal=chanIdxGlobal-elecCounter;

            chanConf=[];
            if ~isempty(chanIdxGlobal)
                for iChan=1:length(chanIdxGlobal)
                    tmp=sq(ieegCARHGZs(chanIdxLocal(iChan),:,51:size(ieegCARHGZs,3)-50));
                    sig2analyzeAllFeature=tmp(:,find(sigMatA.(Task.Conds(iC).Name).(Task.Conds(iC).Field(iF).Name)(chanIdxGlobal(iChan),:)==1));
                    minSize=min(size(sig2analyzeAllFeature));
                    numDim=floor(0.5*minSize);
                    varVect = mean(sig2analyzeAllFeature(:,1:numDim),2);

                    accAll = 0;
                    ytestAll = {};
                    ypredAll = {};
                    if(numFolds>0)
                        cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
                    else
                        cvp = cvpartition(labels,'LeaveOut');
                    end
                    tic
                    parfor nCv = 1:cvp.NumTestSets
                        nCv
                        sig = sig2analyzeAllFeature;
                        labs = labels;
                        train = cvp.training(nCv);
                        test = cvp.test(nCv);
                        ieegTrain = sig(train,:);
                        ieegTest = sig(test,:);
                        %         matTrain = size(ieegTrain);
                        %         gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
                        %         matTest = size(ieegTest);
                        %         gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

                        pTrain = labs(train);
                        pTest = labs(test);
                        gTrain=ieegTrain;
                        gTest=ieegTest;
                        [lossVect,aucVect] = scoreSelect(gTrain,pTrain,varVect,1,numFolds); % Hyper parameter tuning
                        [~,optimDim] = min(mean(lossVect,1)); % Selecting the optimal principal components
                        %        mean(squeeze(aucVect(:,nDim,:)),1)
                        [lossMod,Cmat,yhat,aucVect] = pcaDecode(gTrain,gTest,pTrain,pTest,optimDim,0);
                        ytestAll{nCv} = pTest;
                        ypredAll{nCv} = yhat';
                        % accAll = accAll + 1 - lossMod;
                    end
                    ytestAll = [ytestAll{:}];
                    ypredAll = [ypredAll{:}]';
                    CmatAll = confusionmat(ytestAll,ypredAll);
                    acc = trace(CmatAll)/sum(sum(CmatAll));
                    chanConf(iChan).CmatNorm = CmatAll./sum(CmatAll,2);


                end
            end


            save([DUKEDIR '\Stats\timePerm\Decoding\' Subject(iSN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '_' num2str(numFolds) 'Folds.mat'],'chanConf','chanIdx','chanIdxLocal','chanIdxGlobal');
            elecCounter=elecCounter+size(ieegCARHGZ,1);
        end
    end
end
            