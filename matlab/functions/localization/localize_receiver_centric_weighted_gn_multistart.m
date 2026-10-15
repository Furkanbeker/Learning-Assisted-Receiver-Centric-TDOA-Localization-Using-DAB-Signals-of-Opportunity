function [xHat, info] = localize_receiver_centric_weighted_gn_multistart( ...
    txPos, refPos, deltaTau_s, c, x0, weights, validMask, areaMin, areaMax)
    % Multi-start weighted robust GN solver for receiver-centric TDOA.
    % Why this is needed:
    % - With random TX placement and long line-like UAV trajectories, a single
    %   centroid-based initial point can converge to a wrong hyperbolic branch.
    % - The wrong branch may lie far outside the operating field and cause
    %   GN outliers of several kilometers.
    % - Multi-start selection uses only observable information: estimated TDOA,
    %   PRS weights, field limits, TX/REF geometry, and residuals. It does NOT
    %   use true UAV position, true error, motion type, or labels.

    areaSpan = areaMax - areaMin;
    fieldCenter = [areaMin + 0.5*areaSpan, areaMin + 0.5*areaSpan];

    startList = [];
    startList = add_start_candidate(startList, x0(:).');
    startList = add_start_candidate(startList, mean(txPos,1));
    startList = add_start_candidate(startList, fieldCenter);
    startList = add_start_candidate(startList, refPos(:).');

    % Local starts around the current prediction / previous EKF output.
    % This improves temporal branch consistency without using true position.
    localStep = 0.08 * areaSpan;
    x0Local = x0(:).';
    localOffsets = [ ...
        0 0; 1 0; -1 0; 0 1; 0 -1; ...
        1 1; 1 -1; -1 1; -1 -1] * localStep;
    for iiLocal = 1:size(localOffsets,1)
        pLocal = x0Local + localOffsets(iiLocal,:);
        pLocal(1) = min(max(pLocal(1), areaMin), areaMax);
        pLocal(2) = min(max(pLocal(2), areaMin), areaMax);
        startList = add_start_candidate(startList, pLocal);
    end

    % Coarse field starts. These are cheap compared with PRS processing and
    % strongly reduce wrong-branch convergence for line trajectories.
    g = linspace(areaMin + 0.12*areaSpan, areaMax - 0.12*areaSpan, 5);
    for ix = 1:numel(g)
        for iy = 1:numel(g)
            startList = add_start_candidate(startList, [g(ix), g(iy)]);
        end
    end

    % Add TX-neighborhood and centroid-to-ref midpoint starts. They help when the
    % UAV is close to one side of the TX hull.
    for k = 1:size(txPos,1)
        startList = add_start_candidate(startList, 0.70*txPos(k,:) + 0.30*fieldCenter);
    end
    startList = add_start_candidate(startList, 0.50*mean(txPos,1) + 0.50*refPos(:).');

    nStarts = size(startList,1);
    candidates = repmat(struct( ...
        'x', [NaN NaN], ...
        'score', inf, ...
        'rawRmsNs', inf, ...
        'weightedRmsNs', inf, ...
        'outsideDist', inf, ...
        'start', [NaN NaN], ...
        'singleInfo', struct()), nStarts, 1);

    bestIdx = 1;
    bestScore = inf;

    for sidx = 1:nStarts
        startPoint = startList(sidx,:);

        [candX, candInfo] = localize_receiver_centric_weighted_gn_single( ...
            txPos, refPos, deltaTau_s, c, startPoint, weights, validMask);

        [score, rawRmsNs, weightedRmsNs, outsideDist] = score_gn_candidate( ...
            candX, candInfo, areaMin, areaMax);

        candidates(sidx).x = candX;
        candidates(sidx).score = score;
        candidates(sidx).rawRmsNs = rawRmsNs;
        candidates(sidx).weightedRmsNs = weightedRmsNs;
        candidates(sidx).outsideDist = outsideDist;
        candidates(sidx).start = startPoint;
        candidates(sidx).singleInfo = candInfo;

        if score < bestScore
            bestScore = score;
            bestIdx = sidx;
        end
    end

    % Hard operating-field constraint:
    % The UAV is simulated/operated inside [areaMin, areaMax]^2, so an
    % outside-field GN result is treated as a wrong hyperbolic branch. This uses
    % only known field limits, not true UAV position.
    outsideVals = reshape([candidates.outsideDist], [], 1);
    scoreVals = reshape([candidates.score], [], 1);
    insideIdx = find(isfinite(scoreVals) & outsideVals <= 1e-6);

    if ~isempty(insideIdx)
        [~, locBestInside] = min(scoreVals(insideIdx));
        bestIdx = insideIdx(locBestInside);
    else
        [~, bestIdx] = min(scoreVals + 1000 * outsideVals);
        candidates(bestIdx).x(1) = min(max(candidates(bestIdx).x(1), areaMin), areaMax);
        candidates(bestIdx).x(2) = min(max(candidates(bestIdx).x(2), areaMin), areaMax);
        candidates(bestIdx).outsideDist = 0;
    end

    xHat = candidates(bestIdx).x;
    info = candidates(bestIdx).singleInfo;
    info.multistart_used = true;
    info.multistart_best_idx = bestIdx;
    info.multistart_num_starts = nStarts;
    info.multistart_score = candidates(bestIdx).score;
    info.raw_rms_ns = candidates(bestIdx).rawRmsNs;
    info.weighted_rms_ns = candidates(bestIdx).weightedRmsNs;
    info.outside_field_dist_m = candidates(bestIdx).outsideDist;
    info.best_start = candidates(bestIdx).start;

    % Store compact candidate diagnostics without creating nested tables.
    info.candidate_scores = reshape([candidates.score], [], 1);
    info.candidate_raw_rms_ns = reshape([candidates.rawRmsNs], [], 1);
    info.candidate_outside_dist_m = reshape([candidates.outsideDist], [], 1);
end


% --------------------------------------------------------------------
%  Local helper
%  -------------------------------------------------------------------

function startList = add_start_candidate(startList, p)
    % Append p to the GN start list unless it is invalid or duplicates an
    % existing start (within 1 m).

    p = p(:).';
    if numel(p) ~= 2 || any(~isfinite(p))
        return;
    end

    if isempty(startList)
        startList = p;
        return;
    end

    d = sqrt(sum((startList - p).^2, 2));
    if all(d > 1.0)
        startList = [startList; p]; %#ok<AGROW>
    end
end


% --------------------------------------------------------------------
%  Local helper
%  -------------------------------------------------------------------

function [score, rawRmsNs, weightedRmsNs, outsideDist] = score_gn_candidate( ...
    xHat, gnInfo, areaMin, areaMax)
    % Score a GN candidate (lower is better) from its TDOA residuals and its
    % distance outside the operating field.

    xHat = xHat(:).';

    if any(~isfinite(xHat))
        score = inf;
        rawRmsNs = inf;
        weightedRmsNs = inf;
        outsideDist = inf;
        return;
    end

    if isfield(gnInfo, 'raw_rms_ns') && ~isempty(gnInfo.raw_rms_ns)
        rawRmsNs = gnInfo.raw_rms_ns;
    else
        rawRmsNs = inf;
    end

    if isfield(gnInfo, 'weighted_rms_ns') && ~isempty(gnInfo.weighted_rms_ns)
        weightedRmsNs = gnInfo.weighted_rms_ns;
    else
        weightedRmsNs = rawRmsNs;
    end

    dxLow  = max(areaMin - xHat(1), 0);
    dxHigh = max(xHat(1) - areaMax, 0);
    dyLow  = max(areaMin - xHat(2), 0);
    dyHigh = max(xHat(2) - areaMax, 0);
    outsideDist = hypot(dxLow + dxHigh, dyLow + dyHigh);

    % Residual dominates the score. Field penalty only breaks ties and rejects
    % physically impossible outside-field branches when residuals are similar.
    fieldPenaltyNs = 250.0 * outsideDist;
    score = rawRmsNs + 0.10 * weightedRmsNs + fieldPenaltyNs;
end
