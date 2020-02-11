%
% 1-Dimensional Spectal Estimation via Atomic Norm and Multiple Snapshots
%
% Reference:
%     Y. Li and Y. Chi, "Off-the-Grid Line Spectrum Denoising and 
%     Estimation With Multiple Measurement Vectors," in IEEE Transactions 
%     on Signal Processing, vol. 64, no. 5, pp. 1257-1269, March1, 2016.
%     arXiv:1408.2242
%
% Author: Mo Alloulah
% Date: 100220
%
%%

clear all
close all
clc

%% Synthesize multiple snapshots of a multi-tone signal, randomising phase

N = 256;
L = 8;

sigmaSnr_dB = 40;

nn = [0:N-1].';
f = [-36 5 15];
S = zeros(N, L);
for mm = 1:L
    s = sum(exp( 1i*(2*pi*(f.*nn/N + rand())) ), 2);
    % normalise power level as required
    normPwr_dB = 0;
    pwr_dB = 10*log10(s'*s/length(s));
    sigScale = sqrt(10^(-(pwr_dB-normPwr_dB)/10));
    s = sigScale * s;
    sigPwr_dB = 10*log10(s'*s/length(s));
    % add Gaussian white noise - SNR [dB]
    n_awgn = 10^(-sigmaSnr_dB/20)*(randn(size(s)) + 1i*randn(size(s)))/sqrt(2);
    noisePwr_dB = 10*log10(n_awgn'*n_awgn/length(n_awgn));
    s = s + n_awgn;
    fprintf('SNR = %2.3f (dB)\n', sigPwr_dB-noisePwr_dB);
    S(:, mm) = s;
end

% inspect Fourier spectrum
t_f = nn - N/2;
s_f = fftshift(fft(s));
figure();
plot( t_f, 20*log10( abs(s_f) ) )
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Visualisations

% inspect multiple snapshots
figure();
hold on
for mm = 1:L
    plot( real(S(:, mm)) )
end
hold off

% inspect Fourier spectrum
t_f = nn - N/2;
s_f = fftshift(fft(S));
figure();
plot( t_f, 20*log10( sum(abs(s_f), 2) ) )
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Atomic Norm

% regularizer is a function of SNR
alpha = 8*pi*N*log(N);
tau = 10^(-sigmaSnr_dB/20)*((1 + 1/log(N))^0.5)* ...
    ( L*log(alpha*L) + sqrt(2*L*log(alpha*L)) + sqrt(pi*L/2) + 1 );

Y = S;

% convex optimisation
cvx_begin sdp %quiet

variable X_est(N,L) complex
variable W(L,L)
variable u(1,N) complex

Z = toeplitz(u);

H = [Z X_est;X_est' W];

minimize(sum_square_abs(Y(:)-X_est(:))/2 + (tau/2)*(real(trace(Z)) + real(trace(W))))
subject to
H == hermitian_semidefinite(N+L)

cvx_end

% dual solution
X_hat = Y - X_est;

%% Localise Frequencies

% discrete set of Atoms (Fourier)
A = zeros(N, N);
for f = -N/2:N/2-1
    A(:, 1+f+N/2) = exp(1i*2*pi*f*[0:N-1].'/N); 
end
spectr_an = sum(abs(X_hat'*A), 1);

figure();
plot( t_f, 20*log10( abs(spectr_an) ) )
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Inspect regularisation as a function of SNR

snr_vect = [0:50];
alpha = 8*pi*N*log(N);
tau_vect = 10.^(-snr_vect/20)*((1 + 1/log(N))^0.5)* ...
    ( L*log(alpha*L) + sqrt(2*L*log(alpha*L)) + sqrt(pi*L/2) + 1 );

figure();
plot( snr_vect, tau_vect )
xlim([snr_vect(1) snr_vect(end)])
xlabel('SNR (dB)')
ylabel('\tau')
