%% RMCLS - Delta Sweep (fixed ER window)
%    Kodrasi, Goetze & Doclo, IEEE Trans. Signal Processing, vol. 61, no. 23, 2013.
%
%    Sweeps delta from 0 to 1 to show how the equalized frequency response
%    changes as the tail is progressively weighted in the cost function.
%
%    The ER window is fixed (physically motivated by the RIR structure).
%    Delta is the actual free parameter in RMCLS:
%
%      delta = 0  ->  pMINT: only the window rows matter, tail is ignored
%      delta = 1  ->  full MINT: window and tail are weighted equally
%
%    Unlike MINT_TAU_EIR where increasing tau degrades the FR, here
%    increasing delta transitions the solution from pMINT toward MINT.

%% Load RIR

scriptDir   = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataDir     = fullfile(projectRoot, 'data');

[H, Fs] = audioread(fullfile(dataDir, 'source', 'RIR.wav'));

if size(H, 2) > 1
    H = H(:, 1);
end

peak_h = max(abs(H));
if peak_h < eps
    error('Input RIR appears silent (all zeros). Check data/source/RIR.wav.');
end
H = H / peak_h;
N = length(H);
H = H(:);

%% Common Setup

% Peak detection (direct sound) — shared across all window sizes
[~, peak_idx] = max(abs(H));

% Pre-peak onset — shared across all window sizes
onset_thresh = 0.01;
pre_peak_idx = find(abs(H(1:peak_idx)) > onset_thresh * max(abs(H)), 1, 'first');
if isempty(pre_peak_idx)
    pre_peak_idx = peak_idx;
end

fprintf('Peak index:     %d  (t = %.2f ms)\n', peak_idx, peak_idx/Fs*1000);
fprintf('Pre-peak onset: %d  (t = %.2f ms)\n\n', pre_peak_idx, pre_peak_idx/Fs*1000);

% Target impulse at direct-sound peak (fixed across all sweeps)
D = zeros(N, 1);
D(peak_idx) = 1;

% Full-RIR Toeplitz convolution matrix (built once, reused)
M = N;
r = [H(1); zeros(M-1, 1)];
c = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

freq = (0:N-1) * Fs / N;

%% Delta Sweep (fixed ER window = 50 ms)

early_reflection_ms = 50;   % fixed window — physically motivated

er_end_idx = peak_idx + round(early_reflection_ms / 1000 * Fs);
er_end_idx = min(er_end_idx, N);
win_len    = er_end_idx - pre_peak_idx + 1;

fprintf('ER window:      %d ms  |  %d samples  (indices %d:%d)\n\n', ...
    early_reflection_ms, win_len, pre_peak_idx, er_end_idx);

delta_values   = [0, 0.1, 0.25, 0.5, 0.75, 1];
colors         = lines(length(delta_values));
legend_entries = cell(length(delta_values), 1);

figure;
hold on;
title('EIR Magnitude Response — \delta sweep (RMCLS, ER window = 50 ms)');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;

for k = 1:length(delta_values)
    d = delta_values(k);

    % Build the diagonal of Lambda (the penalty matrix).
    % The cost function is J = ||Lambda*(H_conv*W - D)||^2, so each row i
    % is penalized by lambda_i^2. We store sqrt(delta) so that after squaring
    % inside the norm the tail penalty becomes exactly delta:
    %   inside window  -> lambda_i = 1        -> penalty = 1^2   = 1   (hard)
    %   outside window -> lambda_i = sqrt(d)  -> penalty = d^2   = d   (soft)
    lambda_vec = sqrt(d) * ones(N, 1);
    lambda_vec(pre_peak_idx:er_end_idx) = 1;

    % Apply weights
    H_rmcls = lambda_vec .* H_conv;
    D_rmcls = lambda_vec .* D;

    % Solve — no regularization
    W_k = lsqr(H_rmcls, D_rmcls, 1e-6, 500);

    % Equalized IR and frequency response
    EIR_k    = conv(H, W_k);
    EIR_k    = EIR_k(1:N);
    EIR_k    = EIR_k / max(abs(EIR_k));

    EIR_db        = 20 * log10(abs(fft(EIR_k, N)) + eps);
    EIR_db_smooth = smoothdata(EIR_db(1:N/2), 'gaussian', 200);

    fprintf('delta = %.2f  |  error = %e\n', d, norm(EIR_k - D));

    legend_entries{k} = sprintf('\\delta = %.2f', d);

    semilogx(freq(1:N/2), EIR_db_smooth, 'Color', colors(k,:), 'LineWidth', 1.5);
end

legend(legend_entries, 'Location', 'best');
set(gca, 'XScale', 'log');
xlim([20 Fs/2]);
