function featureNames = keep_numeric_table_variables(T, featureNames)
    % Keep only the numeric or logical columns of table T.
    keep = true(size(featureNames));
    for i = 1:numel(featureNames)
        v = T.(featureNames{i});
        keep(i) = isnumeric(v) || islogical(v);
    end
    featureNames = featureNames(keep);
end
