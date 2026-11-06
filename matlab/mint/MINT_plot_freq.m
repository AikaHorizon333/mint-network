%% MINT — Frequency Response Plot
%    Designs the MINT inverse filter W and plots:
%      |H(f)|   — Room Impulse Response
%      |W(f)|   — MINT inverse filter
%      |EIR(f)| — Equalized IR (H * W, should be ~flat)

%% Load RIR
scriptDir   = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataDir     = fullfile(projectRoot, 'data');

[H, Fs] = audioread(fullfile(dataDir, 'source', 'RIR.wav'));
if size(H, 2) > 1, H = H(:, 1); end
H = H / max(abs(H));
H = H(:);
N = length(H);

%% Design MINT Filter
M = N;
c = [H; zeros(M-1, 1)];
r = [H(1); zeros(M-1, 1)];
H_conv = toeplitz(c, r);
H_conv = H_conv(1:N, :);

D = [1; zeros(N-1, 1)];
W = lsqr(H_conv, D, 1e-6, 500);

EIR = conv(H, W);
EIR = EIR(1:N);

fprintf('Equalization error: %e\n', norm(EIR - D));

%% Frequency Axis
Lfft    = 2^nextpow2(4 * N);
nyq     = Lfft / 2;
freq    = (0:nyq-1) * Fs / Lfft;
f_range = [20, Fs/2];

mag_db = @(x) 20 * log10(abs(fft(x, Lfft)) + eps);
smooth_db = @(x) smoothdata(mag_db(x), 'gaussian', 150);

H_db   = smooth_db(H);
W_db   = smooth_db(W);
EIR_db = smooth_db(EIR);

%% Plot
figure('Name', 'MINT Frequency Response', 'Position', [100 100 900 600]);

semilogx(freq, H_db(1:nyq),   'b',  'LineWidth', 1.5); hold on;
semilogx(freq, W_db(1:nyq),   'r',  'LineWidth', 1.5);
semilogx(freq, EIR_db(1:nyq), 'k',  'LineWidth', 1.5);
yline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);

xlim(f_range);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('MINT — Frequency Response');
legend('RIR  |H(f)|', 'Inverse filter  |W(f)|', 'Equalized IR  |H*W(f)|', ...
       'Location', 'best');
grid on;
