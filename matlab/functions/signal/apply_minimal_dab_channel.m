function [y, chOut] = apply_minimal_dab_channel(txWave, delaySamples, outLen, amp_fspl, noisePower)
    % Minimal channel: fractional timing delay + FSPL amplitude + AWGN only.
    x0 = place_delayed_signal(txWave, delaySamples, outLen);
    x0 = amp_fspl * x0;

    signalPower = mean(abs(x0).^2) + eps;
    noise = sqrt(noisePower/2) * (randn(size(x0)) + 1j*randn(size(x0)));
    y = x0 + noise;

    chOut = struct();
    chOut.channel_type = "LOS";
    chOut.snr_dB = 10*log10(signalPower / (noisePower + eps));
end
