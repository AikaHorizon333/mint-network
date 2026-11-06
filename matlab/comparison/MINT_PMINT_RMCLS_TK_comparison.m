%% MINT-TK vs pMINT-TK vs RMCLS-TK — Tikhonov Regularization Comparison
%
%  All three methods share:
%    - Same RIR (H)
%    - Same filter length M = N
%    - Same target delay (peak_idx)
%    - Same lsqr tolerance and iterations
%    - Same Tikhonov beta_reg scalar
%    - Same Wiener preconditioner (beta_pre)
%
%  Structural differences under comparison:
%    - MINT-TK  : full H, Tikhonov + Wiener preconditioner
%    - pMINT-TK : windowed H_w, Tikhonov (no preconditioner)
%    - RMCLS-TK : full H, row-weighted by delta, Tikhonov + Wiener preconditioner

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
beta_reg            = 1e-6;   % Tikhonov weight (relative to matrix Frobenius norm)
M                   = N;      % filter length

%% Shared Derived Quantities

% Peak detection
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
fprintf('beta_reg:         %.2e\n', beta_reg);
fprintf('delta (RMCLS):    %.2f\n', delta);

% Target impulse at direct-sound peak
D = zeros(N, 1);
D(peak_idx) = 1;

% FFT of H (used for Wiener preconditioner)
H_fft = fft(H, N);
freq  = (0:N-1) * Fs / N;

% Full Toeplitz convolution matrix
r      = [H(1); zeros(M-1, 1)];
c      = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

% Wiener preconditioner (shared by MINT-TK and RMCLS-TK)
beta_pre = 1e-4 * max(abs(H_fft).^2);
M_inv    = @(x, tflag) wiener_precond(H_fft, beta_pre, x, tflag);

%% MINT-TK

beta_mint = beta_reg * norm(H_conv, 'fro')^2 / N;
damp_mint = sqrt(beta_mint);

afun_mint  = @(x, tflag) tk_afun(H_conv, damp_mint, x, tflag);
W_mint_tk  = lsqr(afun_mint, [D; zeros(M, 1)], 1e-6, 1000, M_inv);
EIR_mint_tk = conv(H, W_mint_tk);
EIR_mint_tk = EIR_mint_tk(1:N);
fprintf('MINT-TK  verification error: %e\n', norm(EIR_mint_tk - D));

%% pMINT-TK

% Windowed RIR
win      = zeros(N, 1);
win(pre_peak_idx:er_end_idx) = 1;
H_w      = H .* win;

r_w      = [H_w(1); zeros(M-1, 1)];
c_w      = [H_w;    zeros(M-1, 1)];
H_w_conv = toeplitz(c_w, r_w);
H_w_conv = H_w_conv(1:N, :);

beta_pmint = beta_reg * norm(H_w_conv, 'fro')^2 / N;
damp_pmint = sqrt(beta_pmint);

afun_pmint   = @(x, tflag) tk_afun(H_w_conv, damp_pmint, x, tflag);
W_pmint_tk   = lsqr(afun_pmint, [D; zeros(M, 1)], 1e-6, 1000);
EIR_pmint_tk = conv(H, W_pmint_tk);
EIR_pmint_tk = EIR_pmint_tk(1:N);
fprintf('pMINT-TK verification error: %e\n', norm(EIR_pmint_tk - D));

%% RMCLS-TK

lambda_vec = sqrt(delta) * ones(N, 1);
lambda_vec(pre_peak_idx:er_end_idx) = 1;

H_rmcls = lambda_vec .* H_conv;
D_rmcls = lambda_vec .* D;

beta_rmcls = beta_reg * norm(H_rmcls, 'fro')^2 / N;
damp_rmcls = sqrt(beta_rmcls);

afun_rmcls   = @(x, tflag) tk_afun(H_rmcls, damp_rmcls, x, tflag);
W_rmcls_tk   = lsqr(afun_rmcls, [D_rmcls; zeros(M, 1)], 1e-6, 1000, M_inv);
EIR_rmcls_tk = conv(H, W_rmcls_tk);
EIR_rmcls_tk = EIR_rmcls_tk(1:N);
fprintf('RMCLS-TK verification error: %e\n', norm(EIR_rmcls_tk - D));

%% Normalize EIRs for plotting

EIR_mint_n  = EIR_mint_tk  / max(abs(EIR_mint_tk)  + eps);
EIR_pmint_n = EIR_pmint_tk / max(abs(EIR_pmint_tk) + eps);
EIR_rmcls_n = EIR_rmcls_tk / max(abs(EIR_rmcls_tk) + eps);

%% Plot

figure;

% --- Time domain ---
subplot(2, 1, 1);
plot(EIR_mint_n,  'b',  'LineWidth', 1.2); hold on;
plot(EIR_pmint_n, 'r',  'LineWidth', 1.2);
plot(EIR_rmcls_n, 'Color', [0.1 0.7 0.1], 'LineWidth', 1.2);
xline(peak_idx, '--k', 'Peak', 'LineWidth', 1);
title(sprintf('EIR Comparison (Tikhonov) — Time Domain  [\\beta_{reg}=%.0e]', beta_reg));
xlabel('Samples'); ylabel('Amplitude');
legend('MINT-TK', 'pMINT-TK', sprintf('RMCLS-TK (\\delta=%.2f)', delta), 'Location', 'best');
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
title('EIR Comparison (Tikhonov) — Frequency Response');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('MINT-TK', 'pMINT-TK', sprintf('RMCLS-TK (\\delta=%.2f)', delta), 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Helpers

function y = tk_afun(A, damp, x, tflag)
    if strcmp(tflag, 'notransp')
        y = [A * x; damp * x];
    else
        y = A' * x(1:size(A,1)) + damp * x(size(A,1)+1:end);
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
