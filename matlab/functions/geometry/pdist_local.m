function d = pdist_local(P)
    % Pairwise Euclidean distances between the rows of P (no toolbox needed).

    n = size(P,1);
    d = zeros(n*(n-1)/2,1);
    idx = 0;
    for i = 1:n-1
        for j = i+1:n
            idx = idx + 1;
            d(idx) = norm(P(i,:) - P(j,:));
        end
    end
end
