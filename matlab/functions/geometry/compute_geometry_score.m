function geometry_score = compute_geometry_score( ...
    isUavInside, isRefInside, gdop_like, condition_number, ...
    uavToCentroidDist, refToCentroidDist, equivalentRadius)
    % Heuristic geometry score in [0, 1] combining hull membership of UAV
    % and REF, GDOP-like value, condition number and centroid distances.

    insideScore = 0.5 * double(isUavInside) + 0.5 * double(isRefInside);

    gdopScore = 1 / (1 + gdop_like);

    condScore = 1 / (1 + log10(condition_number + 1));

    scale = max(equivalentRadius, 1);
    uavCentroidScore = exp(-uavToCentroidDist / (2.5 * scale));
    refCentroidScore = exp(-refToCentroidDist / (2.5 * scale));

    centroidScore = 0.5 * uavCentroidScore + 0.5 * refCentroidScore;

    geometry_score = ...
        0.35 * insideScore + ...
        0.30 * gdopScore + ...
        0.20 * condScore + ...
        0.15 * centroidScore;

    geometry_score = max(0, min(1, geometry_score));
end
