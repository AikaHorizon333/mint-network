%% MINT - Tau Sweep (no regularization)
%    Miyoshi & Kaneda, IEEE Trans. Acoustics, Speech & Signal Processing,
%    vol. 36, no. 2, 1988.
%
%    Sweeps tau from 0 to N/2 to show how the equalized frequency
%    response degrades as the target impulse delay increases.
%    No Tikhonov regularization — pure MINT.

%% Load RIR

scriptDir   = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataDir     = fullfile(projectRoot, 'data');

[H, Fs] = audioread(fullfile(dataDir, 'source', 'RIR.wav'));

if size(H, 2) > 1
    H = H(:, 1);
end

H = H / max(abs(H));
N = length(H);
H = H(:);
freq = (0:N-1) * Fs / N;

%% Build Convolution Matrix

M = N;
r = [H(1); zeros(M-1, 1)];
c = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

%% Tau Sweep

tau_values    = round([0, N/8, N/4, 3*N/8, N/2]);
colors        = lines(length(tau_values));
legend_entries = cell(length(tau_values), 1);

figure;
hold on;
title('EIR Magnitude Response — \tau sweep (no regularization)');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;

for k = 1:length(tau_values)
    t = tau_values(k);

    D_t        = zeros(N, 1);
    D_t(t + 1) = 1;

    W_t   = lsqr(H_conv, D_t, 1e-6, 500);
    EIR_t = conv(H, W_t);
    EIR_t = EIR_t(1:N);
    EIR_t = EIR_t / max(abs(EIR_t));

    EIR_db     = 20 * log10(abs(fft(EIR_t, N)) + eps);
    EIR_db_smooth = smoothdata(EIR_db(1:N/2), 'gaussian', 200);

    fprintf('tau = %5d  |  error = %e\n', t, norm(EIR_t - D_t));
    legend_entries{k} = sprintf('\\tau = %d  (%.0f ms)', t, t/Fs*1000);

    semilogx(freq(1:N/2), EIR_db_smooth, 'Color', colors(k,:), 'LineWidth', 1.5);
end

legend(legend_entries, 'Location', 'best');
set(gca, 'XScale', 'log');
xlim([20 Fs/2]);
