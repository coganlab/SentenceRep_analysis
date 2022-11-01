[SMLS idxSM]=intersect(sigChans{1}{1},SM);
[AUDLS idxAUD]=intersect(sigChans{1}{1},AUD);

iiSM=find(sigMatStartTime{1}{1}(idxSM)>=50);
iiAUD=find(sigMatStartTime{1}{1}(idxAUD)>=50);


AUD1=intersect(sigChans{1}{1},sigChans{5}{1}); % LS LM
AUD2=intersect(AUD1,sigChans{3}{1}); % LS/LM JL
PROD1=intersect(sigChans{1}{2},sigChans{5}{2}); % LS LM
SM=intersect(AUD1,PROD1); % AUD PROD
AUD=setdiff(AUD2,SM); 
PROD=setdiff(PROD1,SM);


% these are significant channels that start after aud onset
SM2=intersect(SM,sigChans{1}{1}(idxSM(iiSM)));
AUD2=intersect(AUD,sigChans{1}{1}(idxAUD(iiAUD)));

% 

idx=SM; %1:2451; %sigChans{1}{2};% PROD; %sigChans; %1:size(sigMatA,1);%SMV;
[W,H]=nnmf(sigMatA{1}{1}(idx,:),4);
figure;
plot(H');

[Wmax maxIdx]=max(W,[],2);
W1=find(maxIdx==1);
W2=find(maxIdx==2);
W3=find(maxIdx==3);
W1pos=find(W(:,1)>0);
W2pos=find(W(:,2)>0);
W3pos=find(W(:,3)>0);
W1=intersect(W1pos,W1);
W2=intersect(W2pos,W2);
W3=intersect(W3pos,W3);

W4=find(maxIdx==4);
W4pos=find(W(:,4)>0);
W4=intersect(W4pos,W4);



%%

dVals=[];
dValsConf=[];
globalChansAll=[];
for iSN=1:length(Subject);
   load([Subject(iSN).Name '_SentenceRep_LSwords_AuditorywDelay_Start_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_LSLM_AuditorywDelay_Start_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_LMwords_AuditorywDelay_StartPre_10Folds.mat'])   
 %   load([Subject(iSN).Name '_SentenceRep_LSwords_Response_StartPre_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_LSwords_DelaywGo_StartPre_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_LSsentences_AuditorywDelay_StartPre_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_JLsentences_AuditorywDelay_StartPre_10Folds.mat'])
 %   load([Subject(iSN).Name '_SentenceRep_LSsentences_DelaywGo_StartPre_10Folds.mat'])

    dValsTmp=[];
    dValsConfTmp=[];
    for iChan=1:length(chanIdxLocal);
        dValsTmp(iChan)=mean(diag(chanConf(iChan).CmatNorm));
        dValsConfTmp(iChan,:,:)=chanConf(iChan).CmatNorm;
    end
    globalChansAll=cat(1,globalChansAll,chanIdxGlobal);
    dVals=cat(2,dVals,dValsTmp);
    dValsConf=cat(1,dValsConf,dValsConfTmp);
end

for iChan=1:size(dValsConf,1)
    dValsConfDiag(iChan)=mean(diag(sq(dValsConf(iChan,:,:))));
end

    