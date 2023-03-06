function chanBinAll = loadTaskSubjectData(Task,Subject,task_dir)
    chanBinAll={};
    SNList=1:length(Subject);
    for iSN=1:length(SNList) %Subjects
        SN=SNList(iSN);
        chanIdx=Subject(SN).goodChannels;
        for iC=1:length(Task.Conds) %Conditions
            for iF=1:length(Task.Conds(iC).Field) %Epochs
                statsFile = [task_dir '\Stats\timePerm\' Subject(SN).Name '_' Task.Name '_' ...
                     Task.Conds(iC).Name '_' Task.Conds(iC).Field(iF).Name '_' Task.Base.Name '.mat'];
                 if ~isfile(statsFile)
                     disp([statsFile ' has not yet been created'])
                     continue % make this skip the whole subject
                 end
                load(statsFile, 'chanSig', 'ieegCARHG', 'ieegCARHGZ');          
                
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
                chanBinAll{iSN}{iC}{iF}.Zs = ieegCARHGZ;
            end
        end
        display(iSN)
    end
end