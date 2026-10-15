function a = wrapToPi_local(a)
    % Wrap angles to [-pi, pi) (no Mapping Toolbox needed).

    a = mod(a + pi, 2*pi) - pi;
end
