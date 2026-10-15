function geometry_class = classify_geometry(geometry_score)
    % Map a geometry score to "good" / "medium" / "bad".

    if geometry_score >= 0.60
        geometry_class = "good";
    elseif geometry_score >= 0.35
        geometry_class = "medium";
    else
        geometry_class = "bad";
    end
end
