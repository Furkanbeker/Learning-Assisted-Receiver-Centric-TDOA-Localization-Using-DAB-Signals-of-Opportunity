function [txPos, refPos, uavPath, centerPos, scaleValue, startDeg, clockwise, motionType] = ...
    generate_random_geometry_and_uav_motion(numTx, areaMin, areaMax, minRefToTx, numPts, scenarioIdx)
    % Generate a random TX/REF layout and a UAV trajectory that pass the
    % geometry quality gates below. Motion types are assigned round-robin
    % over the scenario index, so every type gets the same number of
    % scenarios.

    assert(numTx == 4, 'Random dataset geometry expects exactly 4 transmitters.');

    % Motion types used in the dataset. The path generator also supports
    % "am_sine" (sine with a contracting/expanding amplitude envelope); add
    % it to this list to enable it.
    motionList = ["circle", "line", "sine", "sine2"];

    motionType = motionList(mod(scenarioIdx-1, numel(motionList)) + 1);

    areaSpan = areaMax - areaMin;
    fieldMargin = max(80, 0.02 * areaSpan);

    %% Random TX/REF/path placement with quality gates
    % TX positions are drawn uniformly over the field (minus a margin).
    % The following gates only remove degenerate cases such as nearly collinear
    % TX layouts, extremely poor TDOA conditioning, or paths that sit in a
    % geometry where GN is expected to be numerically unstable.
    % No true localization error is used here.

    minTxSeparation = max(450, 0.09 * areaSpan);
    minHullArea = 0.07 * areaSpan^2;
    maxTry = 9000;

    minMeanPathGeomScore = 0.22;
    minWorstPathGeomScore = 0.08;
    maxMeanPathGdop = 4.50;
    maxWorstPathCond = 180;

    txPos = [];
    refPos = [];
    uavPath = [];
    centerPos = [NaN NaN];
    scaleValue = NaN;
    startDeg = NaN;
    clockwise = false;

    for attempt = 1:maxTry

        candTx = areaMin + fieldMargin + ...
            (areaSpan - 2*fieldMargin) * rand(numTx, 2);

        pairD = pdist_local(candTx);
        hullArea = safe_hull_area(candTx);

        if min(pairD) < minTxSeparation || hullArea < minHullArea
            continue;
        end

        candTxGeomInfo = compute_tx_polygon_info(candTx);

        candRef = sample_random_point_far_from_tx( ...
            candTx, areaMin, areaMax, minRefToTx, fieldMargin, maxTry);

        theta = 2*pi*rand;
        u = [cos(theta), sin(theta)];
        v = [-sin(theta), cos(theta)];

        t = linspace(0, 1, numPts).';
        candClockwise = false;
        candStartDeg = rad2deg(theta);

        maxFieldRadius = 0.36 * areaSpan;
        typicalPathLength = min(max(0.48 * areaSpan * (0.80 + 0.40*rand), 1000), 0.78 * areaSpan);
        halfLength0 = 0.5 * typicalPathLength;

        switch lower(char(motionType))

            case 'circle'
                candStartDeg = 360 * rand;
                candClockwise = rand > 0.5;
                radius = min(max(0.085 * areaSpan * (0.70 + 0.70*rand), 260), maxFieldRadius);
                candCenter = sample_center_with_margin(areaMin, areaMax, radius + fieldMargin);
                candPath = generate_circular_path(candCenter, radius, numPts, candStartDeg, candClockwise);
                candScale = radius;

            case 'line'
                [candPath, candCenter, candScale] = generate_random_line_like_path( ...
                    areaMin, areaMax, fieldMargin, u, v, halfLength0, 0, 0, numPts, t);

            case {'sine','sine2','am_sine','sine_am'}
                baseAmplitude = min(max(0.050 * areaSpan * (0.70 + 0.80*rand), 150), 470);

                if strcmpi(char(motionType), 'sine2')
                    cycles = 2.0;
                    baseAmplitude = 0.85 * baseAmplitude;
                    envelopeMode = "constant";
                elseif strcmpi(char(motionType), 'am_sine') || strcmpi(char(motionType), 'sine_am')
                    cycles = 3.0;
                    baseAmplitude = 0.95 * baseAmplitude;
                    envelopeMode = "contract_expand";
                else
                    cycles = 1.0;
                    envelopeMode = "constant";
                end

                [candPath, candCenter, candScale] = generate_random_line_like_path( ...
                    areaMin, areaMax, fieldMargin, u, v, halfLength0, baseAmplitude, cycles, numPts, t, envelopeMode);

            otherwise
                error('Unknown UAV motion type: %s', motionType);
        end

        candPath(:,1) = min(max(candPath(:,1), areaMin + fieldMargin), areaMax - fieldMargin);
        candPath(:,2) = min(max(candPath(:,2), areaMin + fieldMargin), areaMax - fieldMargin);

        geomQuality = evaluate_planned_scenario_geometry( ...
            candTx, candRef, candPath, candTxGeomInfo);

        if geomQuality.meanScore >= minMeanPathGeomScore && ...
           geomQuality.minScore  >= minWorstPathGeomScore && ...
           geomQuality.meanGdop  <= maxMeanPathGdop && ...
           geomQuality.maxCond   <= maxWorstPathCond

            txPos = candTx;
            refPos = candRef;
            uavPath = candPath;
            centerPos = candCenter;
            scaleValue = candScale;
            startDeg = candStartDeg;
            clockwise = candClockwise;
            break;
        end
    end

    if isempty(txPos)
        % Fallback: still random, but only the TX spacing and hull-area
        % gates are applied. This guarantees termination for very unlucky
        % random draws without resorting to a fixed TX layout.
        for attempt = 1:maxTry
            candTx = areaMin + fieldMargin + ...
                (areaSpan - 2*fieldMargin) * rand(numTx, 2);
            pairD = pdist_local(candTx);
            hullArea = safe_hull_area(candTx);
            if min(pairD) >= minTxSeparation && hullArea >= minHullArea
                txPos = candTx;
                break;
            end
        end

        if isempty(txPos)
            txPos = areaMin + fieldMargin + ...
                (areaSpan - 2*fieldMargin) * rand(numTx, 2);
        end

        refPos = sample_random_point_far_from_tx(txPos, areaMin, areaMax, minRefToTx, fieldMargin, maxTry);
        theta = 2*pi*rand;
        u = [cos(theta), sin(theta)];
        v = [-sin(theta), cos(theta)];
        t = linspace(0, 1, numPts).';
        startDeg = rad2deg(theta);
        clockwise = false;
        halfLength0 = 0.34 * areaSpan;
        [uavPath, centerPos, scaleValue] = generate_random_line_like_path( ...
            areaMin, areaMax, fieldMargin, u, v, halfLength0, 0, 0, numPts, t);
    end

    if any(~isfinite(uavPath(:))) || any(~isfinite(refPos(:))) || any(~isfinite(txPos(:)))
        error('Generated non-finite random geometry in scenario %d.', scenarioIdx);
    end

    txGeomInfo = compute_tx_polygon_info(txPos);
    if txGeomInfo.area <= 0
        error('Invalid TX hull area in scenario %d.', scenarioIdx);
    end
end
