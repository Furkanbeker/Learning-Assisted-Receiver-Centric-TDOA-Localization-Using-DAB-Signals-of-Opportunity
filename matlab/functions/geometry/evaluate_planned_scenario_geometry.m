function geomQuality = evaluate_planned_scenario_geometry(txPos, refPos, uavPath, txGeomInfo)
    % Geometry statistics along a candidate UAV path (about 12 sampled
    % points), used by the scenario quality gates.

    refGeomInfo = compute_point_geometry_info(refPos, txPos);
    N = size(uavPath,1);

    scores = zeros(N,1);
    gdops = zeros(N,1);
    conds = zeros(N,1);

    stride = max(1, floor(N / 12));
    idxList = unique([1:stride:N, N]);

    for ii = 1:numel(idxList)
        idx = idxList(ii);
        p = uavPath(idx,:);
        uavGeomInfo = compute_point_geometry_info(p, txPos);
        gdopInfo = compute_tdoa_geometry_metrics(txPos, refPos, p);
        scores(ii) = compute_geometry_score( ...
            uavGeomInfo.isInsideTxPolygon, refGeomInfo.isInsideTxPolygon, ...
            gdopInfo.gdop_like, gdopInfo.condition_number, ...
            uavGeomInfo.pointToCentroidDist, refGeomInfo.pointToCentroidDist, ...
            txGeomInfo.equivalentRadius);
        gdops(ii) = gdopInfo.gdop_like;
        conds(ii) = gdopInfo.condition_number;
    end

    scores = scores(1:numel(idxList));
    gdops = gdops(1:numel(idxList));
    conds = conds(1:numel(idxList));

    geomQuality = struct();
    geomQuality.meanScore = mean(scores);
    geomQuality.minScore = min(scores);
    geomQuality.meanGdop = mean(gdops);
    geomQuality.maxCond = max(conds);
end
