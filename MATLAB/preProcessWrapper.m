% top level wrapper for preProcess workflow

global BOX_DIR
global RECONDIR
%global TASK_DIR
BOX_DIR='C:\Users\ae166\Box';
RECONDIR=([BOX_DIR '\ECoG_Recon']);
path = fullfile(userpath, 'MATLAB-env');ae12
addpath(genpath(path)); % repo at https://github.com/coganlab/MATLAB-env
Task=makeTask();
%Task.Name='Phoneme_Sequencing';%'LexicalDecRepDelay';
Task.Name='SentenceRep';
%Task.Name='LexicalDecRepNoDelay'
%Task.Name='Neighborhood_Sternberg';
%TASK_DIR=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
%Task.Name='LexicalDecRepNoDelay';
%Task.Name='LexicalDecRepDelay';

Task.Directory=([BOX_DIR '/CoganLab/D_Data/' Task.Name]);
Task.Outlier.Field='Auditory'; % this is used to define the field for outlier channel removals
if ~exist('Subjects','var')
    Subjects = popTaskSubjectData(Task);
end

Subject = getSubjects(Subjects,{'D73'});
preProcess_LineFilter(Task,Subject); % -> cleanieeg.dat
preProcess_ChannelOutlierRemoval(Task, Subject)
makeSentenceRepTrials
%preProcess_ResponseCoding(Task,Subjects);
%preProcess_Specgrams

%Sentence_Rep_timeClusterStats.m
%Sentence_Rep_timeClusterStats_NNMF.m

