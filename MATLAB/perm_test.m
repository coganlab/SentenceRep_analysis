load('D103_pythonVal.mat')
load('D103_data.mat')
pchanmat = pChanPy;
base = mean(ieegBaseData, 3);
field = mean(ieegFieldData, 3);
for i =1:length(pchanmat)
pchanmat(i) = permtest(field(i,:), base(i,:), 10000);
end
scatter(pchanmat, pChanPy)
xlabel("MATLAB")
ylabel("python")
title("D103 MATLAB vs python permutation p values")