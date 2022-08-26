function condIdx = sentRepSort(trialInfo)
   % condIdx = 2D condition IDs (trials x ids)
   % condIdx(:,1) - cue ids; 1 - Listen, 2 - :=:
   % condIdx(:,2) - sound ids; 1 - 'heat', 2 - 'hoot', 3 - 'hot', 4 - 'hut'
   %                           5 - 'dog', 6 - 'fame', '7' - 'mice'
   % condIdx(:,3) - Go ids; 1 - 'speak', 2 - 'mime', 3 - empty (should
   %                                                correspond with cue id 2)
   condIdx = zeros(length(trialInfo), 3);

   % Assigning cue ids - Default with ones
    if(iscell(trialInfo))
        for iTrial = 1:length(trialInfo)
            cueAll{iTrial} = trialInfo{iTrial}.cue;
            soundAll{iTrial} = trialInfo{iTrial}.sound;
            goAll{iTrial} = trialInfo{iTrial}.go;
        end
    else
        
        cueAll = {trialInfo(:).cue};
        soundAll = {trialInfo(:).sound};
        goAll = {trialInfo(:).go};
    end
   condIdx(:,1) = ones(length(trialInfo),1);
   
   condIdx(contains(cueAll, ':=:'),1) = 2;

   % Assigning sound ids - Default with ones
   condIdx(:,2) = ones(length(trialInfo),1);
   
   condIdx(contains(soundAll, 'hoot'),2) = 2;
   condIdx(contains(soundAll, 'hot'),2) = 3;
   condIdx(contains(soundAll, 'hut'),2) = 4;
   condIdx(contains(soundAll, 'dog'),2) = 5;
   condIdx(contains(soundAll, 'fame'),2) = 6;
   condIdx(contains(soundAll, 'mice'),2) = 7;

   % Assigning go ids - Default with three for empty cells
   condIdx(:,3) = 3. * ones(length(trialInfo),1);
   
   condIdx(contains(goAll, 'Speak'),3) = 1;
   condIdx(contains(goAll, 'Mime'),3) = 2;       

end