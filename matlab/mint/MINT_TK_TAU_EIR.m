%% MINT w/ Delay 
%    Miyoshi & Kaneda, IEEE Trans. Acoustics, Speech & Signal Processing,
%    vol. 36, no. 2, 1988.
%
%    Designs an inverse filter W (FIR, length M = N) such that:
%
%      H * W ≈ δ(n)      (convolution with room ≈ impulse)
%
%    Formulated as a least-squares problem using a Toeplitz convolution matrix:
%      Where D = [0 ... 0 1 0 ... 0] has a delay
%      min || H_conv * W - D ||^2
%
%    where H_conv is the N x M Toeplitz matrix built from H, and D is the
%    target impulse at sample 0. Solved iteratively with lsqr.
%
%    W contains the FIR filter coefficients.
%    The Equalized IR (EIR = H * W) is saved to data/equalized_ir.wav.

%% Load RIR

% Get project root (parent of matlab/)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataDir = fullfile(projectRoot, 'data');

[H, Fs] = audioread(fullfile(dataDir, 'source', 'RIR.wav'));

if size(H, 2) > 1
    H = H(:, 1);
end

% Normalize the RIR
H = H / max(abs(H));

% Compute frequency response
% - N is the number of samples in H (the RIR)
% - H_fft is the FFT
% - freq are the frequencies. 
% -- They are calculated using the Sample Rate (Fs)
N = length(H);
H_fft = fft(H, N);
freq = (0:N-1) * Fs / N;

% Get magnitude and phase
magnitude = abs(H_fft);
magnitude_db = 20 * log10(magnitude + eps); % Add eps to avoid log(0)
phase = angle(H_fft);

% Plot all in one figure
figure;

% Time domain RIR
subplot(2, 1, 1);
plot(H);
title('RIR - Time Domain');
xlabel('Time (samples)');
ylabel('Amplitude');
grid on;

% Magnitude response (log scale)
subplot(2, 1, 2);
semilogx(freq(1:N/2), magnitude_db(1:N/2));
title('Frequency Response - Magnitude');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
xlim([20 30000]);

%% Design the Filter (MINT Approach)
% MINT: Multiple-input/output INverse-filtering Theorem
% For SISO case: Design inverse filter (W) such that H * W ≈ D (impulse)

% Ensure H is a column vector
H = H(:);

% Peak detection — sets tau to align the target with the direct sound
[~, peak_idx] = max(abs(H));
fprintf('Peak index: %d  (t = %.2f ms)\n', peak_idx, peak_idx/Fs*1000);

% Filter length
M = N;

% Toeplitz convolution matrix
r = [H(1); zeros(M-1, 1)];
c = [H;    zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

% Target impulse at direct-sound peak (consistent with pMINT and RMCLS)
%   tau = peak_idx - 1  ->  D(tau+1) = D(peak_idx) = 1
tau = peak_idx - 1;
D = zeros(N, 1);
D(tau + 1) = 1;

% Tikhonov regularisation via augmented system:
%   [ H_conv ]       [ D ]
%   [ damp*I ] W  =  [ 0 ]
% Stabilises spectral nulls and dramatically improves lsqr convergence.
beta_reg = 1e-6;
beta     = beta_reg * norm(H_conv, 'fro')^2 / N;
damp     = sqrt(beta);

fprintf('Tikhonov damp: %.4e\n', damp);

% Circulant preconditioner (Wiener deconvolution in frequency domain):
%   notransp: M^{-1}  x = ifft( H*(f) * X(f) / (|H(f)|^2 + beta_pre) )
%   transp:  (M^{-T}) x = ifft( H(f)  * X(f) / (|H(f)|^2 + beta_pre) )
% Flattens the condition number of the preconditioned system.
beta_pre = 1e-4 * max(abs(H_fft).^2);
M_inv    = @(x, tflag) wiener_precond(H_fft, beta_pre, x, tflag);

% Base MINT (no regularization)
W_base       = lsqr(H_conv, D, 1e-6, 1000);
EIR_base     = conv(H, W_base);
EIR_base     = EIR_base(1:N);
fprintf('MINT base verification error: %e\n', norm(EIR_base - D));

% MINT-TK (Tikhonov + Wiener preconditioner)
afun = @(x, tflag) mint_tau_afun(H_conv, damp, x, tflag);
W = lsqr(afun, [D; zeros(N, 1)], 1e-6, 500, M_inv);

% Verify the filter on the original RIR (Equalized IR)
H_equalized = conv(H, W);
H_equalized = H_equalized(1:N);
error_norm = norm(H_equalized - D);
fprintf('MINT-TK  verification error: %e\n', error_norm);

% Save the Equalized IR as a WAV file
% H_equalized_norm = H_equalized / max(abs(H_equalized));  % Normalize to [-1, 1] for WAV
% equalized_ir_filename = fullfile(dataDir, 'MINT-TAU-EIR.wav');
% audiowrite(equalized_ir_filename, H_equalized_norm, Fs);
% fprintf('Equalized IR saved to: %s\n', equalized_ir_filename);

% Plot the filter and verification
figure;

% Inverse filter
subplot(2, 2, 1);
plot(W);
title('Inverse Filter (W)');
xlabel('Time (samples)');
ylabel('Amplitude');
grid on;

% Verification: Original RIR convolved with filter
subplot(2, 2, 2);
plot(H_equalized);
hold on;
plot(D, 'r--', 'LineWidth', 1.5);
title('Verification: H * W vs Target (D)');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('H * W', 'Target (impulse)', 'Location', 'best');
grid on;
hold off;

% Zoomed verification (first 100 samples)
subplot(2, 2, 3);
plot(H_equalized(1:min(100, length(H_equalized))));
hold on;
plot(D(1:min(100, length(D))), 'r--', 'LineWidth', 1.5);
title('Verification - Zoomed (First 100 samples)');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('H * W', 'Target (impulse)', 'Location', 'best');
grid on;
hold off;

% Error plot
subplot(2, 2, 4);
error_signal = H_equalized - D;
plot(error_signal);
title('Error: H * W - D');
xlabel('Time (samples)');
ylabel('Error');
grid on;

%% Base vs TK Comparison

EIR_base_n = EIR_base     / max(abs(EIR_base)     + eps);
EIR_tk_n   = H_equalized  / max(abs(H_equalized)  + eps);
freq_plot  = (0:N-1) * Fs / N;

figure;

subplot(2, 1, 1);
plot(EIR_base_n, 'b',  'LineWidth', 1.2); hold on;
plot(EIR_tk_n,   'r',  'LineWidth', 1.2);
xline(peak_idx, '--k', 'Peak', 'LineWidth', 1);
title('MINT: Base vs Tikhonov — Time Domain');
xlabel('Samples'); ylabel('Amplitude');
legend('MINT (no reg)', sprintf('MINT-TK (\\beta_{reg}=%.0e)', beta_reg), 'Location', 'best');
grid on; hold off;

base_db = smoothdata(20*log10(abs(fft(EIR_base_n, N)) + eps), 'gaussian', 200);
tk_db   = smoothdata(20*log10(abs(fft(EIR_tk_n,   N)) + eps), 'gaussian', 200);

subplot(2, 1, 2);
semilogx(freq_plot(1:N/2), base_db(1:N/2), 'b', 'LineWidth', 1.5); hold on;
semilogx(freq_plot(1:N/2), tk_db(1:N/2),   'r', 'LineWidth', 1.5);
title('MINT: Base vs Tikhonov — Frequency Response');
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
legend('MINT (no reg)', sprintf('MINT-TK (\\beta_{reg}=%.0e)', beta_reg), 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Helpers

function y = mint_tau_afun(A, damp, x, tflag)
    if strcmp(tflag, 'notransp')
        y = [A * x; damp * x];
    else
        y = A' * x(1:size(A,1)) + damp * x(size(A,1)+1:end);
    end
end

%% Helper — Wiener circulant preconditioner for lsqr

function y = wiener_precond(H_fft, beta_pre, x, tflag)
    denom = abs(H_fft).^2 + beta_pre;
    if strcmp(tflag, 'notransp')
        y = real(ifft( conj(H_fft) .* fft(x) ./ denom ));
    else
        y = real(ifft( H_fft .* fft(x) ./ denom ));
    end
end
