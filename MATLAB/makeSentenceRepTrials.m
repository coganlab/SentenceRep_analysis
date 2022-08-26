for iTrials=1:length(Trials);
    Trials(iTrials).StartCode=trialInfo{iTrials}.cond;
    Trials(iTrials).AuditoryCode=Trials(iTrials).StartCode+25;
    Trials(iTrials).GoCode=Trials(iTrials).StartCode+50;
end
save('Trials.mat','Trials');
