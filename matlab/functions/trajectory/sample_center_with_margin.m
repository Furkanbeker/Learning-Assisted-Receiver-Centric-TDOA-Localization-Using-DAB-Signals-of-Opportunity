function centerPos = sample_center_with_margin(areaMin, areaMax, requiredMargin)
    % Sample a random point that keeps requiredMargin from the field edges.

    areaSpan = areaMax - areaMin;
    margin = min(requiredMargin, 0.45 * areaSpan);
    lo = areaMin + margin;
    hi = areaMax - margin;

    if hi <= lo
        centerPos = [areaMin + 0.5*areaSpan, areaMin + 0.5*areaSpan];
    else
        centerPos = lo + (hi - lo) * rand(1,2);
    end
end
