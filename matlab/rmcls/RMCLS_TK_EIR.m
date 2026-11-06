%% RMCLS - Relaxed Multichannel Least-Squares (SISO adaptation)
%    Kodrasi, Goetze & Doclo, IEEE Trans. Signal Processing, vol. 61, no. 23, 2013.
%   
%    RMCLS generalises pMINT by introducing a relaxation parameter delta that
%    controls how strictly the equalization is enforced OUTSIDE the direct
%    sound + early-reflection window:
%
%      J_RMCLS(W) = || P  (H*W - D) ||^2          <- window: enforce hard
%                 + || sqrt(delta) * P_perp (H*W - D) ||^2  <- tail: enforce softly
%   
%    delta = 0  ->  pMINT  (tail completely ignored)
%    delta = 1  ->  full MCLS / MINT  (tail weighted equally)
%   
%    Implemented as a row-weighted least-squares solve:
%      Lambda = diag([..., 1 (inside window), ..., sqrt(delta) (outside), ...])
%      min || Lambda*(H_conv*W - D) ||^2  +  damp^2 * ||W||^2

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
    error('Input RIR appears silent (all zeros). Check data/RIR.wav.');
end
H = H / peak_h;
N = length(H);
H = H(:);

%% Windowing — Isolate Direct Sound + Early Reflections

% --- Tunable parameters ---
onset_thresh       = 0.01;   % fraction of peak to detect pre-peak onset
early_reflection_ms = 50;    % ms of early reflections to include after peak
delta              = 0.1;    % relaxation: 0 = ignore tail, 1 = full equalization
beta_reg           = 1e-6;   % Tikhonov regularisation weight (relative to matrix energy)

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

fprintf('Peak index:         %d  (t = %.2f ms)\n', peak_idx, peak_idx/Fs*1000);
fprintf('Pre-peak onset:     %d  (t = %.2f ms)\n', pre_peak_idx, pre_peak_idx/Fs*1000);
fprintf('ER end index:       %d  (t = %.2f ms)\n', er_end_idx, er_end_idx/Fs*1000);
fprintf('Window length:      %d samples (%.2f ms)\n', ...
    er_end_idx - pre_peak_idx + 1, (er_end_idx - pre_peak_idx + 1)/Fs*1000);
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
title('Frequency Response - Original vs Windowed');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('Original H', 'Windowed H\_w', 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Design the Filter (RMCLS)
%
% Uses the FULL RIR convolution matrix (unlike pMINT which uses H_w).
% Row weights lambda_i encode the relaxation:
%   lambda_i = 1          for i in [pre_peak_idx, er_end_idx]  (window)
%   lambda_i = sqrt(delta) for i outside the window             (tail)
%
% The weighted system (Lambda * H_conv) * W = Lambda * D is then solved
% with an additional Tikhonov damp term for spectral-null stabilisation.

M = N;

% Full-RIR Toeplitz convolution matrix
r = [H(1); zeros(M-1, 1)];
c = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

% Target impulse at direct-sound peak (inside window by construction)
D = zeros(N, 1);
D(peak_idx) = 1;

% Row-weight vector (avoid building an explicit N×N diagonal matrix)
lambda_vec = sqrt(delta) * ones(N, 1);
lambda_vec(pre_peak_idx:er_end_idx) = 1;

% Apply weights by scaling rows
H_rmcls = lambda_vec .* H_conv;   % equivalent to Lambda * H_conv
D_rmcls = lambda_vec .* D;        % equivalent to Lambda * D

% Tikhonov regularisation via augmented system (MATLAB lsqr has no damp arg):
%   [ H_rmcls ]       [ D_rmcls ]
%   [ damp*I  ] W  =  [    0    ]
% Function handles avoid materialising the (N+M) x M augmented matrix.
beta = beta_reg * norm(H_rmcls, 'fro')^2 / N;
damp = sqrt(beta);

fprintf('Tikhonov damp:      %.4e\n', damp);

% Circulant preconditioner M^{-1} (Wiener deconvolution in frequency domain):
%   notransp: M^{-1}  x = ifft( H*(f) * X(f) / (|H(f)|^2 + beta_pre) )
%   transp:  (M^{-T}) x = ifft( H(f)  * X(f) / (|H(f)|^2 + beta_pre) )
% Approximates H_conv^{-1} in O(N log N); dramatically reduces lsqr iterations.
% beta_pre prevents division by zero at spectral nulls of H.
beta_pre = 1e-4 * max(abs(H_fft).^2);
M_inv = @(x, tflag) wiener_precond(H_fft, beta_pre, x, tflag);

% Base RMCLS (no regularization)
W_base       = lsqr(H_rmcls, D_rmcls, 1e-6, 1000);
EIR_base     = conv(H, W_base);
EIR_base     = EIR_base(1:N);
fprintf('RMCLS base verification error: %e\n', norm(EIR_base - D));

% RMCLS-TK (Tikhonov + Wiener preconditioner)
afun = @(x, tflag) rmcls_afun(H_rmcls, damp, M, x, tflag);
W = lsqr(afun, [D_rmcls; zeros(M, 1)], 1e-6, 1000, M_inv);

%% Verify Against Full RIR

H_equalized = conv(H, W);
H_equalized = H_equalized(1:N);
D_full = zeros(N, 1);
D_full(peak_idx) = 1;

error_norm = norm(H_equalized - D_full);
fprintf('RMCLS-TK verification error (full H): %e\n', error_norm);


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

%% Base vs TK Comparison

freq_plot  = (0:N-1) * Fs / N;
EIR_base_n = EIR_base    / max(abs(EIR_base)    + eps);
EIR_tk_n   = H_equalized / max(abs(H_equalized) + eps);

figure;

subplot(2, 1, 1);
plot(EIR_base_n, 'b', 'LineWidth', 1.2); hold on;
plot(EIR_tk_n,   'r', 'LineWidth', 1.2);
xline(peak_idx, '--k', 'Peak', 'LineWidth', 1);
title(sprintf('RMCLS: Base vs Tikhonov — Time Domain  (\\delta=%.2f)', delta));
xlabel('Samples'); ylabel('Amplitude');
legend('RMCLS (no reg)', sprintf('RMCLS-TK (\\beta_{reg}=%.0e)', beta_reg), 'Location', 'best');
grid on; hold off;

base_db = smoothdata(20*log10(abs(fft(EIR_base_n, N)) + eps), 'gaussian', 200);
tk_db   = smoothdata(20*log10(abs(fft(EIR_tk_n,   N)) + eps), 'gaussian', 200);

subplot(2, 1, 2);
semilogx(freq_plot(1:N/2), base_db(1:N/2), 'b', 'LineWidth', 1.5); hold on;
semilogx(freq_plot(1:N/2), tk_db(1:N/2),   'r', 'LineWidth', 1.5);
title('RMCLS: Base vs Tikhonov — Frequency Response');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('RMCLS (no reg)', sprintf('RMCLS-TK (\\beta_{reg}=%.0e)', beta_reg), 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Helper — augmented matrix-vector product for lsqr

function y = rmcls_afun(A, damp, M, x, tflag)
    if strcmp(tflag, 'notransp')
        y = [A * x; damp * x];          % [A; damp*I] * x
    else
        y = A' * x(1:size(A,1)) + damp * x(size(A,1)+1:end);  % [A'; damp*I] * x
    end
end

function y = wiener_precond(H_fft, beta_pre, x, tflag)
    denom = abs(H_fft).^2 + beta_pre;
    if strcmp(tflag, 'notransp')
        y = real(ifft( conj(H_fft) .* fft(x) ./ denom ));
    else
        y = real(ifft( H_fft .* fft(x) ./ denom ));
    end
end
