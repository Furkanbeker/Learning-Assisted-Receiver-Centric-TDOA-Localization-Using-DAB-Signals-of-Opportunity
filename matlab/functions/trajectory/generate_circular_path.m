function path = generate_circular_path(centerPos, radius, numPts, startDeg, clockwise)
    % One full circle sampled at numPts points.

    angles = linspace(0, 2*pi, numPts + 1);
    angles(end) = [];

    if clockwise
        angles = -angles;
    end

    angles = angles + deg2rad(startDeg);

    x = centerPos(1) + radius * cos(angles);
    y = centerPos(2) + radius * sin(angles);

    path = [x(:), y(:)];
end
