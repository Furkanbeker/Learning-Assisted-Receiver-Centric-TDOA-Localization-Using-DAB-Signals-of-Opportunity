function [path, centerPos, scaleValue] = generate_random_line_like_path( ...
    areaMin, areaMax, fieldMargin, u, v, halfLength0, amplitude0, cycles, numPts, t, envelopeMode)
    % Generate a straight or sinusoidal path along direction u with lateral
    % offsets along v. amplitude0 = 0 or cycles = 0 gives a straight line.
    % If the path does not fit inside the field, its length and amplitude
    % are shrunk iteratively.

    if nargin < 11
        envelopeMode = "constant";
    end

    areaSpan = areaMax - areaMin;
    halfLength = halfLength0;
    amplitude = amplitude0;

    for rr = 1:180
        marginNeeded = fieldMargin + amplitude + 40;
        centerPos = sample_center_with_margin(areaMin, areaMax, marginNeeded);

        s = linspace(halfLength, -halfLength, numPts).';
        baseLine = centerPos + s.*u;

        if amplitude <= 0 || cycles <= 0
            offset = zeros(numPts,1);
        else
            if strcmpi(char(envelopeMode), 'contract_expand')
                % Large amplitude at the beginning/end, small amplitude near the
                % middle. This visually resembles an AM envelope that narrows and
                % then widens again.
                envelope = 0.22 + 0.78 * abs(2*t - 1);
                envelope = smoothstep01(envelope);
            else
                envelope = ones(numPts,1);
            end
            offset = amplitude * envelope .* sin(2*pi*cycles*t);
        end

        path = baseLine + offset.*v;

        insidePath = all(path(:,1) >= areaMin + fieldMargin) && all(path(:,1) <= areaMax - fieldMargin) && ...
                     all(path(:,2) >= areaMin + fieldMargin) && all(path(:,2) <= areaMax - fieldMargin);
        if insidePath
            scaleValue = max(halfLength, amplitude);
            return;
        end

        halfLength = 0.97 * halfLength;
        amplitude = 0.95 * amplitude;

        if halfLength < 0.12 * areaSpan
            halfLength = 0.12 * areaSpan;
        end
    end

    % Robust fallback: generate a shorter centered path that is guaranteed to
    % remain inside the field after clipping.
    centerPos = [areaMin + 0.5*areaSpan, areaMin + 0.5*areaSpan] + 0.20*areaSpan*(rand(1,2)-0.5);
    halfLength = min(halfLength, 0.25*areaSpan);
    amplitude = min(amplitude, 0.06*areaSpan);
    s = linspace(halfLength, -halfLength, numPts).';
    baseLine = centerPos + s.*u;

    if amplitude <= 0 || cycles <= 0
        offset = zeros(numPts,1);
    else
        if strcmpi(char(envelopeMode), 'contract_expand')
            envelope = 0.22 + 0.78 * abs(2*t - 1);
            envelope = smoothstep01(envelope);
        else
            envelope = ones(numPts,1);
        end
        offset = amplitude * envelope .* sin(2*pi*cycles*t);
    end

    path = baseLine + offset.*v;
    path(:,1) = min(max(path(:,1), areaMin + fieldMargin), areaMax - fieldMargin);
    path(:,2) = min(max(path(:,2), areaMin + fieldMargin), areaMax - fieldMargin);
    centerPos = mean(path,1);
    scaleValue = max(halfLength, amplitude);
end
