% duke;
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

Task.Baseline.Name='Start';
Task.Baseline.Epoch='Start';
Task.Baseline.Time=[-1000 0];




if ~exist('Subject','var')
    Subjects = popTaskSubjectData(Task);
    Subject([25:28, 30:32]) = [];
end



SNList=1:length(Subject);
SNList=20:24;
for iSN=1:length(SNList);
    SN=SNList(iSN);
    chanIdx=Subject(SN).goodChannels;
    for iC=1:length(Task.Conds)
        for iF=1:length(Task.Conds(iC).Field)
            load(fullfile(DUKEDIR, 'Stats', 'timePerm', [Subject(SN).Name '_' Task.Name '_' ...
                Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat']));
            baseTimeRange=Task.Base.Time;
            baseTimeRange(1)=baseTimeRange(1)-500;
            baseTimeRange(2)=baseTimeRange(2)+500;
            TimeRange=Task.Conds(iC).Field(iF).Time;
            TimeRange(1)=TimeRange(1)-500;
            TimeRange(2)=TimeRange(2)+500;
            TimeLength=TimeRange(2)-TimeRange(1);
            
            totChanBlock=ceil(size(ieegCARHG,1)./60);
            iChan2=0;
            chanSigBin=zeros(length(chanSig),1);
            tscale=linspace(TimeRange(1)+500,TimeRange(2)-500,size(ieegCARHG,3)-100);
            sigBinMat=zeros(size(ieegCARHG,1),size(ieegCARHG,3)-100);
            for iG=0:totChanBlock-1;
                FigS=figure('Position', get(0, 'Screensize'));
                % figure
                for iChan=1:min(60,size(ieegCARHG,1)-iChan2);
                    subplot(6,10,iChan);
                    iChan2=iChan+iG*60;
                    sig1=sq(ieegCARHGZ(iChan2,:,51:size(ieegCARHG,3)-50));
                    sig2=repmat(sq(ieegBaseCARHGZ(iChan2,:,51:100)),1,size(sig1,2)./50);
                    % sig2=repmat(sq(ieegCARHG(iChan2,:,51:100)),1,size(sig1,2)./50);
                    
                    plot(tscale,mean(sig1))
                    hold on;
                    plot(tscale,mean(sig2))
                    %
                    for iCl=1:length(chanSig{iChan2}.actClust.Size)
                        if chanSig{iChan2}.actClust.Size{iCl}>chanSig{iChan2}.actClust.perm95
                            chanSigBin(iChan2)=1;
                            sigBinMat(iChan2,chanSig{iChan2}.actClust.Start{iCl}:chanSig{iChan2}.actClust.Start{iCl}...
                                +(chanSig{iChan2}.actClust.Size{iCl}-1))=ones(chanSig{iChan2}.actClust.Size{iCl},1);
                            hold on;
                            plot(tscale(chanSig{iChan2}.actClust.Start{iCl}:chanSig{iChan2}.actClust.Start{iCl}...
                                +(chanSig{iChan2}.actClust.Size{iCl}-1)),... % -1?
                                -0.5*ones(chanSig{iChan2}.actClust.Size{iCl},1),'k-','LineWidth',4);% min(mean(sig1))
                        end
                    end
                    %         hold on;
                    %         plot(tscale,sq(mean(ieegCARHG(iChan2,iiJL,51:350))));
                    axis('tight')
                    ylim([-0.5 1.5])
                    %         title([Subject(SN).ChannelInfo(Subject(SN).goodChannels(iChan2)).Name ' ' ...
                    %             Subject(SN).ChannelInfo(Subject(SN).goodChannels(iChan2)).Location]);
                    %title([Subject(SN).ChannelInfo(Subject(SN).goodChannels(iChan2)).Location]);
                    title([Subject(SN).ChannelInfo(chanIdx(iChan2)).Name]);
                    
                    % caxis([0.7 1.2]);
                    %  caxis([0.7 1.4]);
                    
                end
                supertitle([Subject(SN).Name ' ' Task.Name ' ' Task.Conds(iC).Name ' ' Task.Conds(iC).Field(iF).Name ' ' num2str(iG+1)])
                F=getframe(FigS);
                if ~exist([DUKEDIR '/Figs/' Subject(SN).Name],'dir')
                    mkdir([DUKEDIR '/Figs/' Subject(SN).Name])
                end
                imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_' Task.Name  '_' Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' num2str(iG+1) '.png'],'png');
                %  imwrite(F.cdata,[DUKEDIR '/Figs/' Subject(SN).Name '/' Subject(SN).Name '_NeighborhoodSternberg_Auditory_SpecGrams_200Hz_0.7to1.4C_' num2str(iF+1) '.png'],'png');
                close
            end
        end
    end
end

subj_labels={};
for iChan=1:length(chanIdx);subj_labels{iChan}=Subject(SN).ChannelInfo(chanIdx(iChan)).Name;end;
plot_subj_grouping(subj_labels,chanSigBin,[]);