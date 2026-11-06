%% MINT - Multiple-Input/output INverse-filtering Theorem (SISO)
%    Miyoshi & Kaneda, IEEE Trans. Acoustics, Speech & Signal Processing,
%    vol. 36, no. 2, 1988.
%
%    Designs an inverse filter W (FIR, length M = N) such that:
%
%      H * W ≈ δ(n)      (convolution with room ≈ impulse)
%
%    Formulated as a least-squares problem using a Toeplitz convolution matrix:
%
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

% Filter length (can be adjusted - longer filters may give better results)
% For inverse filtering, filter length should be >= RIR length
M = N;  % Filter length (can be set to N, 2*N, etc. for better conditioning)

% Create convolution matrix from H using Toeplitz structure
% This represents the convolution H * W
% The Toeplitz matrix has H as the first column, shifted versions in subsequent columns
r = [H(1); zeros(M-1, 1)];  % First row: [H(1), 0, 0, ..., 0]
c = [H; zeros(M-1, 1)];      % First column: [H; zeros]
H_conv = toeplitz(c, r);
% Truncate to N rows to match desired output length
H_conv = H_conv(1:N, :);

% Target impulse response (desired output: impulse at time 0)
% Length must match the number of rows in H_conv (N)
% We can even define a delay tau.
D = [1; zeros(N-1, 1)];  % Column vector

% Design the filter using least squares
% Solve: H_conv * W = D
% This finds W such that conv(H, W) ≈ D
W = lsqr(H_conv, D);

% Verify the filter on the original RIR (Equalized IR)
H_equalized = conv(H, W);
H_equalized = H_equalized(1:N);  % Take first N samples
error_norm = norm(H_equalized - D);
fprintf('Filter design verification error: %e\n', error_norm);

% Save the Equalized IR as a WAV file
H_equalized_norm = H_equalized / max(abs(H_equalized));  % Normalize to [-1, 1] for WAV
equalized_ir_filename = fullfile(dataDir, 'equalized_ir.wav');
audiowrite(equalized_ir_filename, H_equalized_norm, Fs);
fprintf('Equalized IR saved to: %s\n', equalized_ir_filename);

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

%% Magnitude response (MINT inverse filter and equalized cascade)
% Zero-pad for a smooth magnitude curve (W is length M; spectrum is interpolated)
Lfft = 2^nextpow2(4 * max(N, M));
W_fft = fft(W, Lfft);
EIR_fft = fft(H_equalized, Lfft);
freq_m = (0:Lfft - 1) * Fs / Lfft;
nyq = floor(Lfft / 2);
mag_W_db = 20 * log10(abs(W_fft(1:nyq)) + eps);
mag_EIR_db = 20 * log10(abs(EIR_fft(1:nyq)) + eps);

figure;
subplot(2, 1, 1);
semilogx(freq_m(1:nyq), mag_W_db);
title('MINT: Inverse Filter Magnitude |W(f)|');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
xlim([20 min(Fs / 2 - 1, 24000)]);

subplot(2, 1, 2);
semilogx(freq_m(1:nyq), mag_EIR_db);
yline(0, 'r--', 'LineWidth', 1);
title('MINT: Equalized IR magnitude |FFT(H*W)| (near 0 dB if convolution matches impulse)');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;
xlim([20 min(Fs / 2 - 1, 24000)]);
