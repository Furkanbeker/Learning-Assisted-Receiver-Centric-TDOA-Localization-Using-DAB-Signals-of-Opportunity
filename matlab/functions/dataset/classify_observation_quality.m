function [score, cls] = classify_observation_quality(meanSnr, meanPeak, meanConf, validCount, numTx)
    % Heuristic observation quality score in [0, 1] from the UAV link SNR,
    % PRS correlation peak, TOA confidence and number of valid TDOAs,
    % and the corresponding class ("good" / "medium" / "bad").
    snrScore = clip01((meanSnr - 5) / 25);
    peakScore = clip01((meanPeak - 0.60) / 0.35);
    confScore = clip01(meanConf);
    validScore = clip01(validCount / max(numTx,1));

    score = 0.35*snrScore + 0.25*peakScore + 0.25*confScore + 0.15*validScore;
    score = clip01(score);

    if score >= 0.70
        cls = "good";
    elseif score >= 0.40
        cls = "medium";
    else
        cls = "bad";
    end
end
