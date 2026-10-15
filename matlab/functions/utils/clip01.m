function y = clip01(x)
    % Clip x to [0, 1].
    y = max(0, min(1, x));
end
