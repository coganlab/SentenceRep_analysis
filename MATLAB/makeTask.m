function Task = makeTask()
Task=[];
Task.Name='SentenceRep';

Task.Base.Name='Start';
Task.Base.Epoch='Start';
Task.Base.Time=[-1000 -500];

%Task.Base.Name='AuditoryPre';
%Task.Base.Epoch='Auditory';
%Task.Base.Time=[-500 0];
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