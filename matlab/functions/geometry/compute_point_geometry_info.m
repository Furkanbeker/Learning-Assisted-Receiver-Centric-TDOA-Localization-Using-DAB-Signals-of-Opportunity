function pointInfo = compute_point_geometry_info(pointPos, txPos)
    % Whether a point lies inside the TX convex hull and its distance to
    % the TX centroid.

    K = convhull(txPos(:,1), txPos(:,2));

    hullX = txPos(K,1);
    hullY = txPos(K,2);

    isInside = inpolygon(pointPos(1), pointPos(2), hullX, hullY);

    centroid = mean(txPos,1);
    pointToCentroidDist = norm(pointPos - centroid);

    pointInfo.isInsideTxPolygon = isInside;
    pointInfo.pointToCentroidDist = pointToCentroidDist;
end
