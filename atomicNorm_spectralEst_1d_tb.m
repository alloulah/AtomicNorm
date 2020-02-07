%
% 1-dimensional Spectal Estimation via Atomic Norm
%
% Based on
%     Bhaskar, Badri Narayan, Gongguo Tang, and Benjamin Recht. "Atomic  
%     norm denoising with applications to line spectral estimation." 
%     IEEE Transactions on Signal Processing 61.23 (2013): 5987-5999.
%     https://arxiv.org/pdf/1204.0562
%
% Author: Mo Alloulah
% Date: 070220
%
%%

clear all
close all
clc

%% Synthesize a multi-tone signal

N = 256;

nn = [0:N-1].';
f = [-36 5 15];
s = sum(exp(1i*2*pi*f.*nn/N), 2)/sqrt(N*length(f));

% add Gaussian white noise - SNR [dB]
sigmaSnr_dB = 40;
s = s + 10^(-sigmaSnr_dB/20)*(randn(size(s)) + 1i*randn(size(s)))/sqrt(2);

% inspect Fourier spectrum
t_f = nn - N/2;
s_f = fftshift(fft(s));
figure();
plot( t_f, 20*log10( abs(s_f) ) )
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Atomic Norm

% regularizer is a function of SNR
tau = 10^(-sigmaSnr_dB/20)*(1 + 1/log(N))*sqrt( N*log(N)*4*pi*log(N) );

y = s;

% convex optimisation
cvx_begin sdp %quiet

variable x_est(N) complex
variable t
variable u(1,N) complex

Z = toeplitz(u);

H = [Z x_est;x_est' t];

minimize(sum_square_abs(y-x_est)/2 + (tau/2)*(real(trace(Z))/N + t))
subject to
H == hermitian_semidefinite(N+1)

cvx_end

% dual solution
z_hat = y - x_est;

%% Localise Frequencies

% discrete set of Atoms (Fourier)
A = zeros(N, N);
for f = -N/2:N/2-1
    A(:, 1+f+N/2) = exp(-1i*2*pi*f*[0:N-1].'/N); 
end
spectr_an = z_hat.'*A;

% should be also equivalent to FFT
spectr_fft = fftshift(fft(z_hat));
figure();
hold on
plot( t_f, 20*log10( abs(spectr_an) ) )
plot( t_f, 20*log10( abs(spectr_fft) ), '--' )
hold off
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Inspect regularisation as a function of SNR

snr_vect = [0:50];
tau_vect = 10.^(-snr_vect/20)*(1 + 1/log(N))*sqrt( N*log(N)*4*pi*log(N) );

figure();
plot( snr_vect, tau_vect )
xlim([snr_vect(1) snr_vect(end)])
xlabel('SNR (dB)')
ylabel('\tau')
