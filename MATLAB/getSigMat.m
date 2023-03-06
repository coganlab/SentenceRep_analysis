function allSigMat = getSigMat(chanBinAll)
    allSigMat = [];
    for iSN = 1:length(chanBinAll)
        allSig = [];
        for iC=1:length(chanBinAll{iSN})
            for iF=1:length(chanBinAll{iSN}{iC})
                ieegCARHGZ = chanBinAll{iSN}{iC}{iF}.Zs;
                allSig(iC,iF,:,:,:) = ieegCARHGZ(:,:,51:size(ieegCARHGZ,3)-50);
            end
        end
%        if iSN = 1
 %       allSigMat = 
    end
end