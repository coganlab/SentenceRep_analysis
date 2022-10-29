function [sigChans, sigMatA, allSigZ, sigMatChansName, sigMatChansLoc] = getChanSigs(Task, Subject, chanBinAll)
    sigMatChansLoc=struct;
    sigMatChansName=struct;
    sigChans=struct;
    sigMatA=struct;
    allSigZ=struct;
    AllSigWMIdx=[];
    
    for iC=1:size(chanBinAll{1},2) 
        cond = Task.Conds(iC).Name;
        for iF=1:size(chanBinAll{1}{iC},2) % the first subject's field length
            allSigZt=[];
            sigMatAt=[];
            sigChanst=[];
            elecCounter=0;
            field = Task.Conds(iC).Field(iF).Name;
            for iSN=1:size(chanBinAll,2)
                if iF > size(chanBinAll{iSN}{iC},2)
                    continue
                end
                sigMatChansCounter=0;
                ieegCARHGZ = chanBinAll{iSN}{iC}{iF}.Zs;
                allSigZt=cat(1,allSigZt,sq(mean(ieegCARHGZ(:,:,51:size(ieegCARHGZ,3)-50),2)));
                sigIdxGC=find(chanBinAll{iSN}{iC}{iF}.sigChans==1);
                % get rid of WM
                [nonWMIdx, nonWMIdxGC]=setdiff(Subject(iSN).goodChannels,Subject(iSN).WM);
                [WMIdx, WMIdxGC]=intersect(Subject(iSN).goodChannels,Subject(iSN).WM);
                chanIdx=intersect(sigIdxGC,nonWMIdxGC);
                AllSigWMIdx=cat(1,AllSigWMIdx,WMIdxGC);
                % chanIdx=setdiff(chanIdx,Subject(iSN).WM);
                %  chanIdx2=
                %sigMat=cat(1,sigMat,chanBinAll{iSN}{iC}{iF}.matrix(chanIdx,:));
               sigMatAt=cat(1,sigMatAt,chanBinAll{iSN}{iC}{iF}.matrix(:,:));
                %  sigMatAll=cat(1,sigMatAll,sigMatA);
               sigChanst=cat(1,sigChanst,chanIdx+elecCounter);
                for iChan=1:length(Subject(iSN).goodChannels)
                    sigMatChansName.(cond).(field){elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Name;
                    sigMatChansLoc.(cond).(field){elecCounter+1}=Subject(iSN).ChannelInfo(Subject(iSN).goodChannels(iChan)).Location;
                    elecCounter=elecCounter+1;
                  %  sigMatChansCounter=sigMatChansCounter+1;
                end
               % elecCounter=elecCounter+size(chanBinAll{iSN}{iC}{iF}.matrix,1);
            end
            allSigZ.(cond).(field)=allSigZt;
            sigMatA.(cond).(field)=sigMatAt;
            sigChans.(cond).(field)=sigChanst;
    
        end
        display(iC)
    end
end