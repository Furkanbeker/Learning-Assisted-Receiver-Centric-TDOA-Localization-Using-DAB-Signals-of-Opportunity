function rawRms = compute_raw_tdoa_rms(txPos, refPos, deltaTau_s, c, x, activeIdx)
    % RMS of the unweighted TDOA residuals (s) at position x.

    r = compute_raw_tdoa_residual(txPos, refPos, deltaTau_s, c, x, activeIdx);
    rawRms = sqrt(mean(r.^2));
end
