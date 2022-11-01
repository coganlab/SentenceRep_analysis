clear; clc
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
    Subjects = popTaskSubjectData(Task);
%     Subject([25:28, 30:32]) = [];
end
Subject = getSubjects(Subjects,'D9');
%%
CondParams = [];
AnalParams = [];
Auditory_nSpec = [];
Go_nSpec = [];
% CondParams.channel = 1;

SNList = [1:length(Subject)]; % D47 (17) for phoneme sequencing not coded yet
for iSN=1:length(SNList)
    SN=SNList(iSN);
    experiment=Subject(SN).Experiment;
%     load([BOX_DIR '/CoganLab/D_Data/' Task.Name '/' Subject(iSN).Name '/mat/experiment.mat']);
%     load([TASK_DIR '/' Subject(iSN).Name '/' Subject(iSN).Date '/mat/Trials.mat'])
%     
%   
%     
%     Subject(SN).Trials=Trials;
    AnalParams.Channel=Subject(SN).goodChannels;
    SelectedChannels=AnalParams.Channel; % really loose: accounts for practice trial confound
    AnalParams.ReferenceChannels = SelectedChannels;
    AnalParams.TrialPooling = 1; %1;  %1; % used to be 1
    AnalParams.dn=0.05;
    AnalParams.Tapers = [0.5 10];
    AnalParams.fk = 200;
    AnalParams.Reference = 'Grand average'; % 'IndRef'; %'Grand average', 'Grand average induced'% induced' 'Single-ended','IndRef';%AnalParams.RefChans=subjRefChansInd(Subject(SN).Name);
    AnalParams.ArtifactThreshold = 12; %8 %12;
    AnalParams.TrialPooling = 0; %1;  %1; % used to be 1
    srate=experiment.recording.sample_rate;
    srate2=srate/4;
    if srate<2048
        AnalParams.pad=2;
    else
        AnalParams.pad=1;
    end
    
    CondParams.conds=2;
    CondParams.Field = Task.Base.Epoch;
    CondParams.bn = Task.Base.Time;
    CondParams.Task = 'SentenceRep';
    
    tic
    [Base_Spec, Base_Data, Base_Trials] = subjSpectrum(Subject(SN), CondParams, AnalParams);
    toc
    
    Condition = Task.Conds(5);
    for iF=1:length(Condition.Field)
        CondParams.Conds=2;
        CondParams.Field = Condition.Field(iF).Epoch;
        CondParams.bn = Condition.Field(iF).Time;
        tic
        [Field_Spec{iF}, Field_Data, Field_Trials{iF}] = subjSpectrum(Subject(SN), CondParams, AnalParams);
%         Field_Data{iF} = Data.IEEG;
        toc
        
    end
    Auditory_Spec = Field_Spec{1};
    Go_Spec = Field_Spec{2};
%% consolidate 


   base=0;
    %base = zeros(1,size(Auditory_Spec{iCode}{iCh},2));
    for iCh = 1:size(Auditory_Spec{1} )
        base=0;
        for iCode = 1:length(Auditory_Spec)
            %base = base + sq(Auditory_Spec{iCode}{iCh}(5,:)); % standard
            %   base= base+mean(sq(Auditory_Spec{iCode}{iCh}(1:10,:)),1); % used to be 1:9
            base= base+mean(sq(Base_Spec{iCode}(iCh,1:10,:)),1); % used to be 1:9
            
            %base2(iCode,:)=std(sq(Auditory_Spec{iCode}{iCh}(1:6,:)),1); % std across time bins?
            
        end
        base = base./length(Auditory_Spec);
        for iCode = 1:length(Auditory_Spec)
            Auditory_nSpec(iCode,:,iCh,:) = squeeze(Auditory_Spec{iCode}(:,iCh,:))./base(ones(1,size(Auditory_Spec{iCode},1)),:);
            Go_nSpec(iCode,iCh,:,:) = squeeze(Go_Spec{iCode}(iCh,:,:))./base(ones(1,size(squeeze(Go_Spec{iCode}(iCh,:,:)),1)),:);
%             Maint_nSpec(iCode,iCh,:,:) = Maint_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Maint_Spec{iCode}{iCh},1)),:);
%             Motor_nSpec(iCode,iCh,:,:) = Motor_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Motor_Spec{iCode}{iCh},1)),:);
%             Start_nSpec(iCode,iCh,:,:) = Start_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Start_Spec{iCode}{iCh},1)),:);
            %  Motor_nSpec(iCode,iCh,:,:) = Motor_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Motor_Spec{iCode}{iCh},1)),:);
            %  Start_nSpec(iCode,iCh,:,:) = Start_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Start_Spec{iCode}{iCh},1)),:);
            % Del_nSpec(iCode,iCh,:,:) = Del_Spec{iCode}{iCh}(:,:)./base(ones(1,size(Del_Spec{iCode}{iCh},1)),:);
            %Auditory_nSpec(iCode,iCh,:,:)=(sq(Auditory_Spec{iCode}{iCh}(:,:))-base2)./(repmat(base,80,1)); % SD LINE
            
        end
        
    end
%% PLotting
totChanBlock=ceil(length(AnalParams.Channel)./60);
iChan2=0;
for iF=0:totChanBlock-1;
    FigS=figure('Position', get(0, 'Screensize'));
    for iChan=1:min(60,length(AnalParams.Channel)-iChan2);
        subplot(6,10,iChan);
        iChan2=iChan+iF*60;
        tvimage(sq((Go_nSpec([1],iChan2,:,1:200))),'XRange',[-0.5,2]);
        title(experiment.channels(AnalParams.Channel(iChan2)).name);
        caxis([0.7 1.2]);
      %  caxis([0.7 1.4]);

    end
    F=getframe(FigS);
    if ~exist([DUKEDIR '/Figs/' Subject(SN).Name],'dir')
        mkdir([DUKEDIR '/Figs/' Subject(SN).Name])
    end
    imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_' Condition.Name '_SpecGrams_' num2str(iF+1) '.png'],'png');    
  %  imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_PhonemeSequence_Auditory_SpecGrams_0.7to1.4C_' num2str(iF+1) '.png'],'png');
    close
end  
end
