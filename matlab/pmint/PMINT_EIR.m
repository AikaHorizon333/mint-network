%% Partial MINT (pMINT) - RIR Filter Design and Processing
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

%% pMINT Windowing - Isolate Early Reflections

% --- Tunable parameters ---
onset_thresh = 0.01;       % fraction of peak used to find pre-peak onset
early_reflection_ms = 50;  % ms after peak to include as early reflections

% Peak detection (direct sound)
[~, peak_idx] = max(abs(H));

% Pre-peak boundary: first sample exceeding onset_thresh of peak amplitude
pre_peak_idx = find(abs(H(1:peak_idx)) > onset_thresh * max(abs(H)), 1, 'first');
if isempty(pre_peak_idx)
    pre_peak_idx = peak_idx;
end

% Early-reflection end boundary
er_end_idx = peak_idx + round(early_reflection_ms / 1000 * Fs);
er_end_idx = min(er_end_idx, N);

fprintf('Peak index:       %d  (t = %.2f ms)\n', peak_idx, peak_idx/Fs*1000);
fprintf('Pre-peak onset:   %d  (t = %.2f ms)\n', pre_peak_idx, pre_peak_idx/Fs*1000);
fprintf('ER end index:     %d  (t = %.2f ms)\n', er_end_idx, er_end_idx/Fs*1000);
fprintf('Window length:    %d samples (%.2f ms)\n', er_end_idx - pre_peak_idx + 1, ...
    (er_end_idx - pre_peak_idx + 1)/Fs*1000);

% Build rectangular window and apply
win = zeros(N, 1);
win(pre_peak_idx:er_end_idx) = 1;
H_w = H .* win;

% Compute frequency responses
H_fft = fft(H, N);
H_w_fft = fft(H_w, N);
freq = (0:N-1) * Fs / N;

magnitude_db = 20 * log10(abs(H_fft) + eps);
magnitude_w_db = 20 * log10(abs(H_w_fft) + eps);

%% Diagnostic plots

figure;

% Original RIR
subplot(3, 1, 1);
plot(H);
title('Original RIR');
xlabel('Time (samples)');
ylabel('Amplitude');
grid on;

% Windowed RIR with boundary markers
subplot(3, 1, 2);
plot(H, 'Color', [0.7 0.7 0.7]); hold on;
plot(H_w, 'r', 'LineWidth', 1.2);
xline(pre_peak_idx, '--g', 'Pre-peak', 'LineWidth', 1.2);
xline(er_end_idx, '--m', 'ER end', 'LineWidth', 1.2);
title('pMINT: Windowed RIR (H\_w)');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('Original H', 'Windowed H\_w', 'Location', 'best');
grid on; hold off;

% Magnitude comparison
subplot(3, 1, 3);
semilogx(freq(1:N/2), magnitude_db(1:N/2), 'Color', [0.7 0.7 0.7]); hold on;
semilogx(freq(1:N/2), magnitude_w_db(1:N/2), 'r');
title('Frequency Response - Original vs Windowed');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('Original H', 'Windowed H\_w', 'Location', 'best');
grid on; hold off;
xlim([20 Fs/2]);

%% Design the Filter (pMINT Approach)
% Partial MINT: build the convolution matrix from the windowed RIR H_w so
% the least-squares system is fully determined (N equations, N unknowns).
% The filter is then verified against the full RIR H.

M = N;

% Convolution matrix built from the WINDOWED RIR H_w
r_w = [H_w(1); zeros(M-1, 1)];
c_w = [H_w; zeros(M-1, 1)];
H_w_conv = toeplitz(c_w, r_w);
H_w_conv = H_w_conv(1:N, :);

% Target: impulse at the direct-sound peak (inside the window by construction)
D = zeros(N, 1);
D(peak_idx) = 1;

W = lsqr(H_w_conv, D, 1e-6, 1000);

% Verify against the FULL original RIR
H_equalized = conv(H, W);
H_equalized = H_equalized(1:N);
error_norm = norm(H_equalized - D);
fprintf('pMINT filter verification error (full H): %e\n', error_norm);

% Also verify against the windowed RIR for comparison
H_w_equalized = conv(H_w, W);
H_w_equalized = H_w_equalized(1:N);
error_norm_w = norm(H_w_equalized - D);
fprintf('pMINT filter verification error (H_w):    %e\n', error_norm_w);

% Save the pMINT Equalized IR
peak_eq = max(abs(H_equalized));
if peak_eq < 1e-10
    warning(['Equalized IR amplitude is near zero; output may sound silent. ', ...
             'Try increasing early_reflection_ms or lowering onset_thresh.']);
    H_equalized_norm = H_equalized;
else
    H_equalized_norm = H_equalized / peak_eq;
end
equalized_ir_filename = fullfile(dataDir, 'pMINT-EIR.wav');
audiowrite(equalized_ir_filename, H_equalized_norm, Fs);
fprintf('pMINT Equalized IR saved to: %s\n', equalized_ir_filename);


%% Results

figure;

% Inverse filter
subplot(2, 2, 1);
plot(W);
title('pMINT Inverse Filter (W)');
xlabel('Time (samples)');
ylabel('Amplitude');
grid on;

% Verification against full H
subplot(2, 2, 2);
plot(H_equalized); hold on;
plot(D, 'r--', 'LineWidth', 1.5);
title('Full H * W vs Target (D)');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('H * W', 'Target', 'Location', 'best');
grid on; hold off;

% Zoomed verification
subplot(2, 2, 3);
n_zoom = min(100, length(H_equalized));
plot(H_equalized(1:n_zoom)); hold on;
plot(D(1:n_zoom), 'r--', 'LineWidth', 1.5);
title('Zoomed (First 100 samples)');
xlabel('Time (samples)');
ylabel('Amplitude');
legend('H * W', 'Target', 'Location', 'best');
grid on; hold off;

% Error plot
subplot(2, 2, 4);
plot(H_equalized - D);
title('Error: H * W - D');
xlabel('Time (samples)');
ylabel('Error');
grid on;

