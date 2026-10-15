function summary = make_empty_summary_row()
    % Return a scenario-summary struct initialized to NaN.
    varNames = get_summary_variable_names();
    summary = struct();
    for i = 1:numel(varNames)
        summary.(varNames{i}) = NaN;
    end
    summary.motion_type = "";
end
