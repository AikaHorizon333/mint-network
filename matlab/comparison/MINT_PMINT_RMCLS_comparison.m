%% MINT vs pMINT vs RMCLS — Fair Comparison
%
%  All three methods share:
%    - Same RIR (H)
%    - Same filter length M = N
%    - Same target delay tau (samples) — set explicitly below
%    - Same lsqr tolerance and max iterations
%    - Same Tikhonov regularization parameter beta
%
%  Structural differences under comparison:
%    - MINT  : full H convolution matrix
%    - pMINT : windowed H_w convolution matrix (early reflections only)
%    - RMCLS : full H convolution matrix, row-weighted by delta
%
%  NOTE: The target delay tau is set to peak_idx (direct-sound peak) so that
%  the design is causally achievable for all three methods.  Using tau = 1
%  (canonical zero-delay MINT) would be ill-conditioned because H is near-zero
%  before the direct sound.

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

%% Shared Parameters

onset_thresh        = 0.01;   % fraction of peak to detect pre-peak onset
early_reflection_ms = 50;     % ms of early reflections after peak
delta               = 0.5;    % RMCLS relaxation: 0 = pMINT-like, 1 = MINT-like
beta                = 1e-3;   % Tikhonov regularization (0 = no reg, increase if EIR noisy)
lsqr_tol            = 1e-6;   % lsqr convergence tolerance
lsqr_maxit          = 3000;   % lsqr max iterations (must be >> 1000 for N~20k)
M                   = N;      % filter length

%% Shared Derived Quantities

% Peak detection — direct sound
[~, peak_idx] = max(abs(H));

% Pre-peak onset
pre_peak_idx = find(abs(H(1:peak_idx)) > onset_thresh * max(abs(H)), 1, 'first');
if isempty(pre_peak_idx)
    pre_peak_idx = peak_idx;
end

% Early-reflection end boundary
er_end_idx = peak_idx + round(early_reflection_ms / 1000 * Fs);
er_end_idx = min(er_end_idx, N);

fprintf('Peak index:       %d  (t = %.2f ms)\n', peak_idx,     peak_idx/Fs*1000);
fprintf('Pre-peak onset:   %d  (t = %.2f ms)\n', pre_peak_idx, pre_peak_idx/Fs*1000);
fprintf('ER end index:     %d  (t = %.2f ms)\n', er_end_idx,   er_end_idx/Fs*1000);
fprintf('delta (RMCLS):    %.2f\n', delta);
fprintf('beta (Tikhonov):  %.2e\n', beta);

% Target delay: direct-sound peak.
% This is the shared design parameter for all three methods.
% tau must be >= peak_idx for a causally achievable inverse filter.
tau = peak_idx;

D = zeros(N, 1);
D(tau) = 1;

% Full Toeplitz convolution matrix
r      = [H(1); zeros(M-1, 1)];
c      = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

%% MINT
% Augment with Tikhonov regularization: solve [H_conv; sqrt(beta)*I] * W = [D; 0]
H_mint_aug = [H_conv; sqrt(beta) * eye(M)];
D_mint_aug = [D;      zeros(M, 1)];

[W_mint, ~, relres_mint, iter_mint] = lsqr(H_mint_aug, D_mint_aug, lsqr_tol, lsqr_maxit);
EIR_mint  = conv(H, W_mint);
EIR_mint  = EIR_mint(1:N);
fprintf('MINT  — lsqr: relres=%.2e  iters=%d  verify_err=%e\n', ...
    relres_mint, iter_mint, norm(EIR_mint - D));

%% pMINT

% Windowed RIR
win      = zeros(N, 1);
win(pre_peak_idx:er_end_idx) = 1;
H_w      = H .* win;

% Convolution matrix from windowed RIR
r_w      = [H_w(1); zeros(M-1, 1)];
c_w      = [H_w;    zeros(M-1, 1)];
H_w_conv = toeplitz(c_w, r_w);
H_w_conv = H_w_conv(1:N, :);

% Augment pMINT design system with same Tikhonov term for fair comparison
H_pmint_aug = [H_w_conv; sqrt(beta) * eye(M)];
D_pmint_aug = [D;        zeros(M, 1)];

[W_pmint, ~, relres_pmint, iter_pmint] = lsqr(H_pmint_aug, D_pmint_aug, lsqr_tol, lsqr_maxit);
EIR_pmint = conv(H, W_pmint);
EIR_pmint = EIR_pmint(1:N);
% Verify against windowed H_w (design check) and full H (performance check)
EIR_pmint_w = conv(H_w, W_pmint);
EIR_pmint_w = EIR_pmint_w(1:N);
fprintf('pMINT — lsqr: relres=%.2e  iters=%d\n', relres_pmint, iter_pmint);
fprintf('pMINT — design error (H_w * W): %e  |  perf error (H * W): %e\n', ...
    norm(EIR_pmint_w - D), norm(EIR_pmint - D));

%% RMCLS

lambda_vec = sqrt(delta) * ones(N, 1);
lambda_vec(pre_peak_idx:er_end_idx) = 1;

H_rmcls = lambda_vec .* H_conv;
D_rmcls = lambda_vec .* D;

% Augment RMCLS system with same Tikhonov term
H_rmcls_aug = [H_rmcls; sqrt(beta) * eye(M)];
D_rmcls_aug = [D_rmcls; zeros(M, 1)];

[W_rmcls, ~, relres_rmcls, iter_rmcls] = lsqr(H_rmcls_aug, D_rmcls_aug, lsqr_tol, lsqr_maxit);
EIR_rmcls = conv(H, W_rmcls);
EIR_rmcls = EIR_rmcls(1:N);
fprintf('RMCLS — lsqr: relres=%.2e  iters=%d  verify_err=%e\n', ...
    relres_rmcls, iter_rmcls, norm(EIR_rmcls - D));

%% Normalize EIRs for plotting

EIR_mint_n  = EIR_mint  / max(abs(EIR_mint)  + eps);
EIR_pmint_n = EIR_pmint / max(abs(EIR_pmint) + eps);
EIR_rmcls_n = EIR_rmcls / max(abs(EIR_rmcls) + eps);

%% Frequency axis

freq = (0:N-1) * Fs / N;

%% Plot

figure;

% --- Time domain ---
subplot(2, 1, 1);
plot(EIR_mint_n,  'b',  'LineWidth', 1.2); hold on;
plot(EIR_pmint_n, 'r',  'LineWidth', 1.2);
plot(EIR_rmcls_n, 'Color', [0.1 0.7 0.1], 'LineWidth', 1.2);
xline(tau, '--k', sprintf('\\tau=%d', tau), 'LineWidth', 1);
title(sprintf('EIR Comparison — Time Domain  (\\tau = %d, \\beta = %.0e)', tau, beta));
xlabel('Samples'); ylabel('Amplitude');
legend('MINT', 'pMINT', sprintf('RMCLS (\\delta=%.2f)', delta), 'Location', 'best');
grid on; hold off;

% --- Frequency domain ---
MINT_db  = 20 * log10(abs(fft(EIR_mint_n,  N)) + eps);
PMINT_db = 20 * log10(abs(fft(EIR_pmint_n, N)) + eps);
RMCLS_db = 20 * log10(abs(fft(EIR_rmcls_n, N)) + eps);

MINT_db_s  = smoothdata(MINT_db(1:N/2),  'gaussian', 200);
PMINT_db_s = smoothdata(PMINT_db(1:N/2), 'gaussian', 200);
RMCLS_db_s = smoothdata(RMCLS_db(1:N/2), 'gaussian', 200);

subplot(2, 1, 2);
semilogx(freq(1:N/2), MINT_db_s,  'b',  'LineWidth', 1.5); hold on;
semilogx(freq(1:N/2), PMINT_db_s, 'r',  'LineWidth', 1.5);
semilogx(freq(1:N/2), RMCLS_db_s, 'Color', [0.1 0.7 0.1], 'LineWidth', 1.5);
title('EIR Comparison — Frequency Response');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('MINT', 'pMINT', sprintf('RMCLS (\\delta=%.2f)', delta), 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);
