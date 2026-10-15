function [x, P] = ct_ekf_predict_only_step(x, P, dt)
    % CT-EKF prediction without a measurement update (used when the GN fix
    % is rejected).

    [xPred, F] = ct_predict_with_jacobian(x, dt);

    Q = diag([ ...
        8^2, ...
        8^2, ...
        4^2, ...
        deg2rad(8)^2, ...
        0.10^2 ...
    ]);

    x = xPred;
    P = F * P * F.' + Q;

    % Controlled covariance inflation during rejected GN frames. This allows
    % the next reliable GN observation to pull the EKF back without letting one
    % catastrophic GN estimate destroy the track.
    P(1:2,1:2) = P(1:2,1:2) + 35^2 * eye(2);

    x(3) = max(0, min(150, x(3)));
    x(5) = max(-0.8, min(0.8, x(5)));
    x(4) = wrapToPi_local(x(4));
end
