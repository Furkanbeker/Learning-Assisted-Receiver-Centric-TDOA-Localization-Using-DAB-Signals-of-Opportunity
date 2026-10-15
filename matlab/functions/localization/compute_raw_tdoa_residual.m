function r = compute_raw_tdoa_residual(txPos, refPos, deltaTau_s, c, x, activeIdx)
    % Unweighted receiver-centric TDOA residuals (s) at position x.

    x = x(:);
    h = zeros(numel(activeIdx),1);
    for ii = 1:numel(activeIdx)
        k = activeIdx(ii);
        sk = txPos(k,:).';
        dk_uav = norm(x - sk);
        dk_ref = norm(refPos(:) - sk);
        dk_uav = max(dk_uav, 1e-12);
        h(ii) = (dk_uav - dk_ref) / c;
    end

    y = deltaTau_s(activeIdx);
    r = y(:) - h(:);
end
