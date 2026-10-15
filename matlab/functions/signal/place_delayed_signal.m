function y = place_delayed_signal(x, delaySamples, outLen)
    % Delay x by a (fractional) number of samples using linear
    % interpolation and return the first outLen samples.

    x = x(:);
    nOut = (0:outLen-1).';
    nIn  = (0:numel(x)-1).';

    y = interp1(nIn, x, nOut - delaySamples, 'linear', 0);
    y(isnan(y)) = 0;
end
