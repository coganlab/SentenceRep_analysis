function subjects = getSubjects(obj,names)
    subs = {};
    if ischar(names)
        names = string(names);
    end
    fields = fieldnames(obj);
    for iSN = 1:length(names)
        name = names{iSN};
        subs{iSN} = getASubject(obj,name);
    end
    subjects = cell2mat(subs);
end
function obj = getASubject(obj,name)
    idx = cellfun(@(x) any(strcmp(x, name)),{obj.Name});
    obj = obj(idx);
end
