function alphaEKF = compute_adaptive_ekf_alpha(maneuverScore, measurementQuality, geometryScore, ekfToGnDist)
    % Blending weight between the CT-EKF and GN positions:
    % reported position = alphaEKF * EKF + (1 - alphaEKF) * GN.

    maneuverScore = clip01(maneuverScore);
    measurementQuality = clip01(measurementQuality);
    geometryScore = clip01(geometryScore);

    distPenalty = clip01((ekfToGnDist - 8) / 30);
    qualityBonus = clip01((measurementQuality - 0.55) / 0.45);
    geometryBonus = clip01((geometryScore - 0.45) / 0.55);

    % alphaEKF close to 1 means mostly EKF; close to 0 means mostly GN.
    % The blend becomes GN-heavy whenever the GN-only track shows
    % maneuvering or the EKF prediction drifts away from GN, so that the
    % filter does not lag behind real curvature (e.g., sine2).
    alphaEKF = 0.90 ...
        - 0.42 * maneuverScore ...
        - 0.28 * distPenalty ...
        + 0.05 * qualityBonus ...
        + 0.04 * geometryBonus;

    % Low-maneuver case: preserve smoothing for line/circle when EKF and GN are
    % already consistent. This keeps the good cases from being weakened.
    if maneuverScore < 0.15 && ekfToGnDist < 14 && measurementQuality > 0.45
        alphaEKF = max(alphaEKF, 0.88);
    end

    % Maneuver case: allow GN to dominate so CT-EKF does not lag through fast
    % direction/amplitude changes.
    if maneuverScore > 0.30 && measurementQuality > 0.50
        alphaEKF = min(alphaEKF, 0.62);
    end

    if maneuverScore > 0.55 && measurementQuality > 0.50
        alphaEKF = min(alphaEKF, 0.48);
    end

    % If EKF and GN disagree strongly while the PRS quality is high, this is
    % usually model lag rather than a GN failure after the multi-start solver.
    if ekfToGnDist > 28 && measurementQuality > 0.60
        alphaEKF = min(alphaEKF, 0.45);
    end

    if ekfToGnDist > 45 && measurementQuality > 0.65
        alphaEKF = min(alphaEKF, 0.35);
    end

    if measurementQuality < 0.35
        alphaEKF = max(alphaEKF, 0.82);
    end

    alphaEKF = min(max(alphaEKF, 0.25), 0.97);
end
