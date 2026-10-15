function row = make_empty_dataset_row()
    % Return a dataset-row struct with every column initialized to NaN
    % (string columns initialized to empty strings).
    varNames = get_dataset_variable_names();
    row = struct();
    for i = 1:numel(varNames)
        row.(varNames{i}) = NaN;
    end
    row.motion_type = "";
    row.geometry_class_est = "";
    row.geometry_class_true = "";
    row.observation_quality_class = "";
    row.master_waveform_id = "";
end
