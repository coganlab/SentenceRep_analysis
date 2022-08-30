function makeSentenceRepTrials(Subject)
global BOX_DIR
Trials = Subject.Trials;
trialInfo = Subject.trialInfo;
for iTrials=1:length(Trials)
    Trials(iTrials).StartCode=trialInfo{iTrials}.cond;
    Trials(iTrials).AuditoryCode=Trials(iTrials).StartCode+25;
    Trials(iTrials).GoCode=Trials(iTrials).StartCode+50;
end
save(fullfile(BOX_DIR, 'CoganLab', 'D_data', 'SentenceRep', ...
    Subject.Name, Subject.Date, 'mat', 'Trials.mat'),'Trials');
