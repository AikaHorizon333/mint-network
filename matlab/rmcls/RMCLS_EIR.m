%% RMCLS - Relaxed Multichannel Least-Squares (SISO adaptation)
%    Kodrasi, Goetze & Doclo, IEEE Trans. Signal Processing, vol. 61, no. 23, 2013.
%
%    RMCLS generalises pMINT by introducing a relaxation parameter delta that
%    controls how strictly the equalization is enforced OUTSIDE the direct
%    sound + early-reflection window:
%
%      J_RMCLS(W) = || P      (H*W - D) ||^2          <- window: enforce hard
%                 + || sqrt(delta) * P_perp (H*W - D) ||^2  <- tail: relax
%
%    delta = 0  ->  pMINT  (tail completely ignored)
%    delta = 1  ->  full MCLS / MINT  (tail weighted equally)
%
%    Implemented as a row-weighted least-squares solve (no regularization):
%      Lambda = diag([..., 1 (inside window), ..., sqrt(delta) (outside), ...])
%      min || Lambda*(H_conv*W - D) ||^2

%% Load RIR

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataDir = fullfile(projectRoot, 'data');

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

%% Windowing — Isolate Direct Sound + Early Reflections

% --- Tunable parameters ---
onset_thresh        = 0.01;   % fraction of peak to detect pre-peak onset
early_reflection_ms = 50;     % ms of early reflections to include after peak
delta               = 0;      % relaxation: 0 = pMINT (ignore tail), 1 = full MINT

% Peak detection (direct sound)
[~, peak_idx] = max(abs(H));

% Pre-peak onset: first sample exceeding onset_thresh of peak
pre_peak_idx = find(abs(H(1:peak_idx)) > onset_thresh * max(abs(H)), 1, 'first');
if isempty(pre_peak_idx)
    pre_peak_idx = peak_idx;
end

% Early-reflection end boundary
er_end_idx = peak_idx + round(early_reflection_ms / 1000 * Fs);
er_end_idx = min(er_end_idx, N);
win_len    = er_end_idx - pre_peak_idx + 1;

fprintf('Peak index:         %d  (t = %.2f ms)\n', peak_idx, peak_idx/Fs*1000);
fprintf('Pre-peak onset:     %d  (t = %.2f ms)\n', pre_peak_idx, pre_peak_idx/Fs*1000);
fprintf('ER window:          %d ms  |  %d samples  (indices %d:%d)\n', ...
    early_reflection_ms, win_len, pre_peak_idx, er_end_idx);
fprintf('delta (relaxation): %.4f\n', delta);

% Build rectangular window for diagnostic plots
win = zeros(N, 1);
win(pre_peak_idx:er_end_idx) = 1;
H_w = H .* win;

%% Diagnostic plots

H_fft   = fft(H,   N);
H_w_fft = fft(H_w, N);
freq    = (0:N-1) * Fs / N;

figure;

subplot(3, 1, 1);
plot(H);
title('Original RIR');
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;

subplot(3, 1, 2);
plot(H, 'Color', [0.7 0.7 0.7]); hold on;
plot(H_w, 'r', 'LineWidth', 1.2);
xline(pre_peak_idx, '--g', 'Pre-peak', 'LineWidth', 1.2);
xline(er_end_idx,   '--m', 'ER end',   'LineWidth', 1.2);
title('RMCLS: Early-Reflection Window');
xlabel('Time (samples)'); ylabel('Amplitude');
legend('Original H', 'Windowed H\_w', 'Location', 'best');
grid on; hold off;

subplot(3, 1, 3);
semilogx(freq(1:N/2), 20*log10(abs(H_fft(1:N/2))   + eps), 'Color', [0.7 0.7 0.7]); hold on;
semilogx(freq(1:N/2), 20*log10(abs(H_w_fft(1:N/2)) + eps), 'r');
title('Frequency Response — Original vs Windowed');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('Original H', 'Windowed H\_w', 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Design the Filter (RMCLS — no regularization)
%
% Uses the FULL RIR convolution matrix.
% Row weights lambda_i encode the relaxation:
%   lambda_i = 1             for i in [pre_peak_idx, er_end_idx]  (window)
%   lambda_i = sqrt(delta)   for i outside the window             (tail)
%
% The weighted system is solved with lsqr (no Tikhonov term).

M = N;

% Full-RIR Toeplitz convolution matrix
r = [H(1); zeros(M-1, 1)];
c = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

% Target impulse at direct-sound peak (inside window by construction)
D = zeros(N, 1);
D(peak_idx) = 1;

% Build the diagonal of Lambda (the penalty matrix).
% The cost function is J = ||Lambda*(H_conv*W - D)||^2, so each row i
% is penalized by lambda_i^2. We store sqrt(delta) so that after squaring
% inside the norm the tail penalty becomes exactly delta:
%   inside window  -> lambda_i = 1            -> penalty = 1^2     = 1     (hard)
%   outside window -> lambda_i = sqrt(delta)  -> penalty = delta^2 = delta (soft)
lambda_vec = sqrt(delta) * ones(N, 1);
lambda_vec(pre_peak_idx:er_end_idx) = 1;

% Apply weights by scaling rows
H_rmcls = lambda_vec .* H_conv;   % equivalent to Lambda * H_conv
D_rmcls = lambda_vec .* D;        % equivalent to Lambda * D

% Solve directly — no augmented system, no damp
W = lsqr(H_rmcls, D_rmcls, 1e-6, 1000);

%% Verify Against Full RIR

H_equalized = conv(H, W);
H_equalized = H_equalized(1:N);
D_full = zeros(N, 1);
D_full(peak_idx) = 1;

error_norm = norm(H_equalized - D_full);
fprintf('RMCLS verification error (full H): %e\n', error_norm);

%% Save EIR

peak_eq = max(abs(H_equalized));
if peak_eq < 1e-10
    warning(['Equalized IR amplitude near zero. ', ...
             'Try increasing delta, early_reflection_ms, or check RIR.']);
    H_equalized_norm = H_equalized;
else
    H_equalized_norm = H_equalized / peak_eq;
end

equalized_ir_filename = fullfile(dataDir, 'RMCLS-EIR.wav');
audiowrite(equalized_ir_filename, H_equalized_norm, Fs);
fprintf('RMCLS Equalized IR saved to: %s\n', equalized_ir_filename);


%% Results

figure;

subplot(2, 2, 1);
plot(W);
title(sprintf('RMCLS Inverse Filter W  (\\delta = %.3f)', delta));
xlabel('Time (samples)'); ylabel('Amplitude'); grid on;

subplot(2, 2, 2);
plot(H_equalized); hold on;
plot(D_full, 'r--', 'LineWidth', 1.5);
title('Full H * W vs Target');
xlabel('Time (samples)'); ylabel('Amplitude');
legend('H * W', 'Target', 'Location', 'best');
grid on; hold off;

subplot(2, 2, 3);
n_zoom = min(200, length(H_equalized));
plot(H_equalized(1:n_zoom)); hold on;
plot(D_full(1:n_zoom), 'r--', 'LineWidth', 1.5);
title('Zoomed (First 200 samples)');
xlabel('Time (samples)'); ylabel('Amplitude');
legend('H * W', 'Target', 'Location', 'best');
grid on; hold off;

subplot(2, 2, 4);
plot(H_equalized - D_full);
title('Error: H * W - D');
xlabel('Time (samples)'); ylabel('Error'); grid on;
