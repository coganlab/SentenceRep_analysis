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
path = fullfile('..', '..','IEEG_Pipelines','MATLAB');
addpath(genpath(path));
OUT_DIR = fullfile(DUKEDIR, 'Stats', 'timePerm');

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
% Task.Conds(1).Field(3).Name='Response';
% Task.Conds(1).Field(3).Epoch='ResponseStart';
% Task.Conds(1).Field(3).Time=[-1 1];

% Task.Conds(2).Name='LSsentences';
% Task.Conds(2).Field(1).Name='AuditorywDelay';
% Task.Conds(2).Field(1).Epoch='Auditory';
% Task.Conds(2).Field(1).Time=[-0.5 5];
% Task.Conds(2).Field(2).Name='DelaywGo';
% Task.Conds(2).Field(2).Epoch='Go';
% Task.Conds(2).Field(2).Time=[-0.5 4];
% Task.Conds(2).Field(3).Name='Response';
% Task.Conds(2).Field(3).Epoch='ResponseStart';
% Task.Conds(2).Field(3).Time=[-1 4];

Task.Conds(3).Name='JLwords';
Task.Conds(3).Field(1).Name='AuditorywDelay';
Task.Conds(3).Field(1).Epoch='Auditory';
Task.Conds(3).Field(1).Time=[-0.5 1.5];
Task.Conds(3).Field(2).Name='DelaywGo';
Task.Conds(3).Field(2).Epoch='Go';
Task.Conds(3).Field(2).Time=[-0.5 1.5];

timePad = 0.5;

if ~exist('Subject','var')
    Subject = popTaskSubjectData(Task);
end

%%
for iSubject=[1]
     
    Subject(iSubject).Name
    Trials=Subject(iSubject).Trials;
    counterN=0;
    counterNR=0;
    noiseIdx=0;
    noResponseIdx=0;
    
    chanIdx=Subject(iSubject).goodChannels;
    %chanIdx = 1:length(Subject(iSubject).ChannelInfo);
    anatName = {Subject(iSubject).ChannelInfo.Location};
    noNameIds = (cellfun(@isempty,anatName))
    if(~isempty(find(noNameIds, 1)))
        save(fullfile(DUKEDIR,Subject(iSubject).Name,'noNameChannels.mat'),"noNameIds")
    end
    anatName = anatName(chanIdx);
    anatName(cellfun(@isempty,anatName)) = {'noName'};
    
%     sensorymotorChan = contains(anatName,'central');
%     whiteChan = contains(anatName,'white');
%     ifgChan = contains(anatName,'opercularis');
%     frontalChan = contains(anatName,'front');
%     temporalChan = contains(anatName,'temporal');
    anatChanId = ones(1,length(anatName));
    disp(['Number of anatomical channels : ' num2str(length(anatChanId))]);
    
    % prodDelayElecs is loaded from the mat file in 'forKumar'
    % Come up with automated way to parse necessary channels
    
    channelName = {Subject(iSubject).Experiment.channels.name};
    channelName = channelName(chanIdx);
    channelName(cellfun(@isempty,channelName)) = {'noName'};
    
%     %muscleChannelsName = channelName(muscleChannels);
%     [~,delayChan] = intersect(channelName,elecNameProductionClean); 
   
%     chan2select = intersect(find(anatChanId),delayChan);
    chan2select = find(anatChanId);
    %chan2select = muscleChannels;
     if(isempty(chan2select))
        disp('No channels found; Iterating next subject');
        continue;
        % Forces the iteration for next subject;
    end
    chanNameSubject = channelName(chan2select);

%%
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
   
    condIdx=ones(length(Subject(iSubject).Trials),1);
    
    baseEpoch=Task.Base.Epoch;
    baseTimeRange=Task.Base.Time; 
    baseTimeExtract = [baseTimeRange(1)-timePad baseTimeRange(2)+timePad];
    Trials=Subject(iSubject).Trials(setdiff(1:length(Subject(iSubject).Trials),noiseIdx));
    
    ieegBase=trialIEEG(Trials,chanIdx,baseEpoch,baseTimeExtract.*1000);
    ieegBase = permute(ieegBase,[2,1,3]);
    fs = Subject(iSubject).Experiment.processing.ieeg.sample_rate;   
    ieegBaseStruct = ieegStructClass(ieegBase, fs, baseTimeExtract, [1 fs/2], baseEpoch);
    clear ieegBase;
    ieegBaseCAR=extractCar(ieegBaseStruct);
    clear ieegBaseStruct;
    [NumTrials,goodtrials] = remove_bad_trials(ieegBaseCAR.data,10);
    goodTrialsCommon = extractCommonTrials(goodtrials);
    waveSpecBase = getWaveletScalogram(ieegBaseCAR.data(chan2select,goodTrialsCommon,:),ieegBaseCAR.fs,fHigh=1000);
    meanFreqChanOut = extractSpecNorm(waveSpecBase.spec,baseTimeExtract,baseTimeRange);
    clear waveSpecBase;
    clear ieegBaseCar;
    for iCond=1:length(Task.Conds)
        trials2Select = setdiff(find(condIdx==iCond),cat(2,noiseIdx,noResponseIdx));
        if(isempty(trials2Select))
            continue;
        end
        % Iterting through field: 'Auditory', 'Delay', 'Response'
        for iField=1:length(Task.Conds(iCond).Field)
            Trials = Subject(iSubject).Trials(trials2Select);
            Epoch=Task.Conds(iCond).Field(iField).Epoch;
            fieldTimeRange=Task.Conds(iCond).Field(iField).Time;          
            fieldTimeExtract = [fieldTimeRange(1)-timePad fieldTimeRange(2)+timePad];            
            ieegField=trialIEEG(Trials,chanIdx,Epoch,fieldTimeExtract.*1000);
            ieegField = permute(ieegField,[2,1,3]);
            ieegFieldStruct = ieegStructClass(ieegField, fs, fieldTimeExtract, [1 fs/2], Epoch);
            clear ieegField
            % Common average referencing
            ieegFieldCAR = extractCar(ieegFieldStruct);
            clear ieegFieldStruct;
            [NumTrials,goodtrials] = remove_bad_trials(ieegFieldCAR.data,10);
            goodTrialsCommon = extractCommonTrials(goodtrials);
            data2spec = ieegFieldCAR.data(chan2select,goodTrialsCommon,:);
            fs = ieegFieldCAR.fs;
            %waveSpecField = getWaveletScalogram(ieegFieldCAR.data(chan2select,goodTrialsCommon,:),ieegFieldCAR.fs,fHigh=1000);
            clear ieegFieldCar
            %             for iChan = 1:length(chan2select)
%                 figure;
% 
%                 subplot(2,1,1);
%                 timeVect = linspace(ieegFieldCAR.tw(1),ieegFieldCAR.tw(2),size(ieegFieldCAR.data,3));
%                 
%                 plot(timeVect,squeeze(ieegFieldCAR.data(chan2select(iChan),goodTrialsCommon,:)),'color',[0 0 0] +0.75);
%                 hold on;
%                 plot(timeVect,mean(squeeze(ieegFieldCAR.data(chan2select(iChan),goodTrialsCommon,:)),1),'color',[0 0 0]);
%                 xlim(fieldTimeRange);
%                
% 
%                 subplot(2,1,2);
%                 specChanMap(waveSpecField.spec,[],1:length(chan2select),[],timeExtract,[-1.5 -1],[-0.25 0.25],[1 500],[70 150],[-2 2],iChan,meanFreqChanOut);
%                 set(gca,'YTick',1:4:length(waveSpecField.fscale));
%                 set(gca,'YTickLabels',waveSpecField.fscale(1:4:end));
%                 xlim(fieldTimeRange);
%                 xlabel(['Time from ' Epoch ' onset (s)'])
%                 ylabel('Frequency (Hz)')
%                 title(chanNameSubject(iChan))
%             end

            totChanBlock=ceil(length(chan2select)./60);
        iChan2=0;
        for iF=0:totChanBlock-1;
            FigS=figure('Position', get(0, 'Screensize'));
            for iChan=1:min(60,length(chan2select)-iChan2);
                subplot(6,10,iChan);
                iChan2=iChan+iF*60;
                iChan2              
                waveSpecField = getWaveletScalogram(data2spec(iChan2,:,:),fs,fHigh=1000);
                %tvimage(sq((Auditory_nSpec([1],iChan2,:,1:200))),'XRange',[-0.5,1]);
                specChanMap(waveSpecField.spec,[],1,[],fieldTimeExtract,[-1.5 -1],[-0.25 0.25],[1 500],[70 150],[-2 2],1,meanFreqChanOut(iChan2,:));
                set(gca,'YTick',1:10:length(waveSpecField.fscale));
                set(gca,'YTickLabels',round(waveSpecField.fscale(1:10:end)));
                set(gca,'FontSize',10);
                xlim(fieldTimeRange);
                title(chanNameSubject(iChan2))
                clear waveSpecField
            end;
            F=getframe(FigS);
            if ~exist([DUKEDIR '/Figs/Wavelet/' Subject(iSubject).Name],'dir')
                mkdir([DUKEDIR '/Figs/Wavelet/' Subject(iSubject).Name])
            end
            imwrite(F.cdata,[DUKEDIR '/Figs/Wavelet/' Subject(iSubject).Name '/' Subject(iSubject).Name '_SentenceRep_' Task.Conds(iCond).Field(iField).Name '_SpecGrams_1000Hz_' num2str(iF+1) '.png'],'png');    
          %  imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_PhonemeSequence_Auditory_SpecGrams_0.7to1.4C_' num2str(iF+1) '.png'],'png');
            close
        end  
        
        
        end
        
    end
end
