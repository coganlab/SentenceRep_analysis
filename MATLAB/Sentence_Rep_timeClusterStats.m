clear; clc
global BOX_DIR
global RECONDIR
global DUKEDIR
BOX_DIR='C:\Users\ae166\Box';
RECONDIR=[BOX_DIR '/ECoG_Recon'];
DUKEDIR = [BOX_DIR '/CoganLab/D_Data/SentenceRep'];
Task=[];
Task.Name='SentenceRep';
% 
% Task.Base.Name='AuditorywDelay';
% Task.Base.Epoch='Auditory';
% Task.Base.Time=[0 1400];

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-1000 -500];
% % 5 conds
Task.Conds(1).Name='LSwords';
Task.Conds(1).Field(1).Name='Delay';
Task.Conds(1).Field(1).Epoch='Auditory';
Task.Conds(1).Field(1).Time=[0 1400];
Task.Conds(1).Field(1).Name='AuditorywDelay';
Task.Conds(1).Field(1).Epoch='Auditory';
Task.Conds(1).Field(1).Time=[-500 1500];
Task.Conds(1).Field(2).Name='DelaywGo';
Task.Conds(1).Field(2).Epoch='Go';
Task.Conds(1).Field(2).Time=[-500 1500];
Task.Conds(1).Field(3).Name='AuditorytoResponse';
Task.Conds(1).Field(3).Epoch='Auditory';
Task.Conds(1).Field(3).Time=[-1000 3000];
Task.Conds(1).Field(4).Name='Response';
Task.Conds(1).Field(4).Epoch='ResponseStart';
Task.Conds(1).Field(4).Time=[-1000 1000];
% 
Task.Conds(2).Name='LSsentences';
Task.Conds(2).Field = [];
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
Task.Conds(3).Field = [];
Task.Conds(3).Field(1).Name='AuditorywDelay';
Task.Conds(3).Field(1).Epoch='Auditory';
Task.Conds(3).Field(1).Time=[-500 1500];
Task.Conds(3).Field(2).Name='DelaywGo';
Task.Conds(3).Field(2).Epoch='Go';
Task.Conds(3).Field(2).Time=[-500 1500];
Task.Conds(3).Field(3).Name='AuditorytoResponse';
Task.Conds(3).Field(3).Epoch='Auditory';
Task.Conds(3).Field(3).Time=[-1000 3000];

Task.Conds(4).Name='JLsentences';
Task.Conds(4).Field = [];
Task.Conds(4).Field(1).Name='AuditorywDelay';
Task.Conds(4).Field(1).Epoch='Auditory';
Task.Conds(4).Field(1).Time=[-500 5000];
Task.Conds(4).Field(2).Name='DelaywGo';
Task.Conds(4).Field(2).Epoch='Go';
Task.Conds(4).Field(2).Time=[-500 4000];

Task.Conds(5).Name='LMwords';
Task.Conds(5).Field = [];
Task.Conds(5).Field(1).Name='AuditorywDelay';
Task.Conds(5).Field(1).Epoch='Auditory';
Task.Conds(5).Field(1).Time=[-500 1500];
Task.Conds(5).Field(2).Name='DelaywGo';
Task.Conds(5).Field(2).Epoch='Go';
Task.Conds(5).Field(2).Time=[-500 1500];
Task.Conds(5).Field(3).Name='AuditorytoResponse';
Task.Conds(5).Field(3).Epoch='Auditory';
Task.Conds(5).Field(3).Time=[-1000 3000];


if ~exist('Subject','var')
    Subjects = popTaskSubjectData(Task);
    Subjects(27) = [];
end
%SNList=26:33;
Subject = getSubjects(Subjects,{ 'D73' });

SNList=1:length(Subject);
%%
for iSN=1:length(SNList)
    SN=SNList(iSN);
    [condIdx, noiseIdx, noResponseIdx]=SentenceRepConds(Subject(SN).Trials);
    %chanIdx=setdiff(Subject(SN).goodChannels,Subject(SN).WM);
    chanIdx=Subject(SN).goodChannels;
    baseEpoch=Task.Base.Epoch;
    baseTimeRange=Task.Base.Time;
    baseTimeRange(1)=baseTimeRange(1)-500;
    baseTimeRange(2)=baseTimeRange(2)+500;
    % To make the base only be LS condition
    notLSwordsIdx = find(condIdx~=5);
    Trials=Subject(SN).Trials(setdiff(1:length(Subject(SN).Trials),cat(2,noiseIdx,notLSwordsIdx')));
    ieegBase=trialIEEG(Trials,chanIdx,baseEpoch,baseTimeRange);
    sample_rate = Subject(SN).Experiment.recording.sample_rate;
    for iC=1:length(Task.Conds)
        for iF=1:length(Task.Conds(iC).Field)
            display(Task.Conds(iC).Field(iF).Name)
            if iC<=2 % remove noisy and no response trials
            Trials=Subject(SN).Trials(setdiff(find(condIdx==iC),cat(2,noiseIdx,noResponseIdx)));
            else
            Trials=Subject(SN).Trials(setdiff(find(condIdx==iC),noiseIdx));
            end
%             if iC == 1
%                 Trials=Subject(SN).Trials(setdiff(find(condIdx==3),cat(2,noiseIdx)));
%             elseif iC == 2
%                 Trials=Subject(SN).Trials(setdiff(find(condIdx==5),cat(2,noiseIdx)));
%             end
            Epoch=Task.Conds(iC).Field(iF).Epoch;
            TimeRange=Task.Conds(iC).Field(iF).Time;
            TimeRange(1)=TimeRange(1)-500;
            TimeRange(2)=TimeRange(2)+500;
            ieeg=trialIEEG(Trials,chanIdx,Epoch,TimeRange);

            [ieegCARHG,ieegBaseCARHG,ieegCARHGZ,ieegBaseCARHGZ] = getCARHG(ieeg, ieegBase, TimeRange, baseTimeRange, [70 150], sample_rate);
            signal = ieegCARHG(:,:,51:size(ieegCARHG,3)-50); % removing the first and last 500 ms
            baseSignal = ieegBaseCARHG(:,:,51:size(ieegBaseCARHG,3)-50); % removing the first and last 500 ms
            chanSig = getSigs(signal,baseSignal, {Subject(SN).ChannelInfo(chanIdx).Name});
            
%             save([DUKEDIR '\Stats\timePerm\' Subject(SN).Name '_' Task.Name '_' ...
%                 Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat'],'chanSig','ieegCARHG','ieegBaseCARHG','ieegCARHGZ','ieegBaseCARHGZ');
              save([DUKEDIR '\Stats\timePerm\' Subject(SN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat'],'chanSig','ieegCARHG','ieegBaseCARHG','ieegCARHGZ','ieegBaseCARHGZ');
              clear chanSig ieegCARHG ieegBaseCARHG ieegCARHGZ ieegBaseCARHGZ
        end
    end
end


function [ieegCARHG,ieegBaseCARHG,ieegCARHGZ,ieegBaseCARHGZ] = getCARHG(ieeg, ieegBase, TimeRange, baseTimeRange, freqrange, sample_rate) 
    ieegCAR=ieeg-mean(ieeg,2);
    ieegBaseCAR=ieegBase-mean(ieegBase,2);
    TimeLength=TimeRange(2)-TimeRange(1);
    TimeLengthBase=baseTimeRange(2)-baseTimeRange(1);
    ieegCARHGZ=zeros(size(ieegCAR,2),size(ieegCAR,1),TimeLength./10);
    ieegBaseCARHGZ=zeros(size(ieegBaseCAR,2),size(ieegBaseCAR,1),TimeLengthBase./10);
    ieegCARHG=zeros(size(ieegCAR,2),size(ieegCAR,1),TimeLength./10);
    ieegBaseCARHG=zeros(size(ieegBaseCAR,2),size(ieegBaseCAR,1),TimeLengthBase./10);
    parfor iChan=1:size(ieegCAR,2)
        [~,hgsig] = EcogExtractHighGammaTrial(sq(ieegCAR(:,iChan,:)), sample_rate, 100, freqrange,TimeRange,TimeRange,[]);
        [~,hgbase] = EcogExtractHighGammaTrial(sq(ieegBaseCAR(:,iChan,:)), sample_rate, 100, freqrange,baseTimeRange,baseTimeRange,[]);

        [m, s]=normfit(reshape(log(hgbase(:,51:100)),size(hgbase,1)*length(51:100),1));
        ieegCARHGZ(iChan,:,:)=(log(hgsig)-m)./s;
        ieegBaseCARHGZ(iChan,:,:)=(log(hgbase)-m)./s;
        ieegCARHG(iChan,:,:)=hgsig;
        ieegBaseCARHG(iChan,:,:)=hgbase;
        display(iChan);
    end
end    

function chanSig = getSigs(ieeg,ieegBase, name)
% Takes two signals and cluster permutes them to isolate significant
% clusters of signal through all channels.
% ieeg is the common average referenced and high gamma filtered signal
% that is being checked for significant signal
% ieegBase is the baseline common average referenced and high gamma
% filtered signal being used as a reference to compare the fist signal to
% name is the set of channel names
    chanSig=cell(1,size(ieeg,1));
    mFactor = size(ieeg,3)/size(ieegBase,3);
    if floor(mFactor)~=mFactor
        error("signal 1 size should be a multiple of signal 2 size\nCurrent ratio is %d",mFactor)
    end
    parfor iChan=1:size(ieeg,1)
        sig1=sq(ieeg(iChan,:,:));
        sig2=repmat(sq(ieegBase(iChan,:,:)),1,mFactor);
        % sIdx=randperm(300);
        % sig2=sig2(:,sIdx);
        %tic
        [zValsRawAct, pValsRaw, actClust]=timePermCluster(sig1,sig2,1000,1,1.645);
        %toc
        chanSig{iChan}.zValsRawAct=zValsRawAct;
        chanSig{iChan}.pValsRaw=pValsRaw;
        chanSig{iChan}.actClust=actClust;
        chanSig{iChan}.Name=name{iChan};%{Subject(SN).ChannelInfo.Name}
        display(iChan)
    end
end




