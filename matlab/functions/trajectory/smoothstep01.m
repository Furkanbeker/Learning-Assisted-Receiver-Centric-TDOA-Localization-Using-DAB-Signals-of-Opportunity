function y = smoothstep01(x)
    % Cubic smoothstep on [0, 1].

    x = max(0, min(1, x));
    y = x.^2 .* (3 - 2*x);
end
