duke;
global BOX_DIR
global RECONDIR
global DUKEDIR
BOX_DIR='H:\Box Sync';
RECONDIR=[BOX_DIR '\ECoG_Recon'];
DUKEDIR = [BOX_DIR '\CoganLab\D_Data\SentenceRep'];
Task=[];
Task.Name='SentenceRep';

% Task.Base.Name='Start';
% Task.Base.Epoch='Start';
% Task.Base.Time=[-1000 -500];

Task.Base.Name='AuditoryPre';
Task.Base.Epoch='Auditory';
Task.Base.Time=[-500 0];
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






Subject = popTaskSubjectData(Task.Name);


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
                sigMatChansName{iC}{iF}{elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Name;
                sigMatChansLoc{iC}{iF}{elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Location;
                elecCounter=elecCounter+1;
                %  sigMatChansCounter=sigMatChansCounter+1;
            end
            % elecCounter=elecCounter+size(chanBinAll{iSN}{iC}{iF}.matrix,1);
        end
        allSigZ{iC}{iF}=allSigZt;
        sigMatA{iC}{iF}=sigMatAt;
        sigChans{iC}{iF}=sigChanst;
        
    end
    display(iC)
end

sigMatStartTime={};
sigMatLength={};
for iC=1:size(chanBinAll{1},2)
    for iF=1:size(chanBinAll{iSN}{iC},2)
        for iChan=1:size(sigChans{iC}{iF},1)
            ii=find(sigMatA{iC}{iF}(sigChans{iC}{iF}(iChan),:)==1);
            sigMatStartTime{iC}{iF}(iChan)=ii(1);
            sigMatLength{iC}{iF}(iChan)=length(ii);
        end
    end
end



AUD1=intersect(sigChans{1}{1},sigChans{5}{1}); % LS LM
AUD2=intersect(AUD1,sigChans{3}{1}); % LS/LM JL
PROD1=intersect(sigChans{1}{2},sigChans{5}{2}); % LS LM
SM=intersect(AUD1,PROD1); % AUD PROD
AUD=setdiff(AUD2,SM);
PROD=setdiff(PROD1,SM);





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
            