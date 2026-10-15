function code = motion_type_to_code(motionType)
    % Map a motion-type name to an integer code (0 = unknown).
    s = lower(strtrim(char(string(motionType))));
    switch s
        case 'circle'
            code = 1;
        case 'line'
            code = 2;
        case 'sine'
            code = 3;
        case 'sine2'
            code = 4;
        case {'am_sine','sine_am'}
            code = 5;
        otherwise
            code = 0;
    end
end
