function txGeomInfo = compute_tx_polygon_info(txPos)
    % Convex hull, area, centroid and equivalent radius of the TX layout.

    K = convhull(txPos(:,1), txPos(:,2));

    hullX = txPos(K,1);
    hullY = txPos(K,2);

    area = polyarea(hullX, hullY);
    centroid = mean(txPos,1);

    equivalentRadius = sqrt(max(area, eps) / pi);

    txGeomInfo.hullIndex = K;
    txGeomInfo.area = area;
    txGeomInfo.centroid = centroid;
    txGeomInfo.equivalentRadius = equivalentRadius;
end
