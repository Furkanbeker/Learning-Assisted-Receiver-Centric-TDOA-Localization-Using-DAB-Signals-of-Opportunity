function m = weighted_median(x, w)
    % Weighted median of x with non-negative weights w.
    x = x(:);
    w = w(:);
    w(~isfinite(w) | w < 0) = 0;
    if isempty(x) || sum(w) <= eps
        m = median(x);
        return;
    end
    [xs, idx] = sort(x);
    ws = w(idx);
    cw = cumsum(ws) / sum(ws);
    k = find(cw >= 0.5, 1, 'first');
    m = xs(k);
end
