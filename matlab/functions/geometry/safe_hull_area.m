function a = safe_hull_area(P)
    % Convex hull area of the points in P; returns 0 for degenerate sets.

    try
        K = convhull(P(:,1), P(:,2));
        a = polyarea(P(K,1), P(K,2));
    catch
        a = 0;
    end
end
