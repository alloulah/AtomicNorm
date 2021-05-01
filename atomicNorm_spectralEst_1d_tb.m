%
% 1-Dimensional Spectal Estimation via Atomic Norm
%
% Reference:
%     Bhaskar, Badri Narayan, Gongguo Tang, and Benjamin Recht. "Atomic  
%     norm denoising with applications to line spectral estimation." 
%     IEEE Transactions on Signal Processing 61.23 (2013): 5987-5999.
%     arXiv:1204.0562
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
% f = [-36 36];   % multi-tone
f = [36];       % single-tone
s = sum(exp(1i*2*pi*f.*nn/N), 2);


% inspect Fourier spectrum
t_f = nn - N/2;
spectr_fft = fftshift(fft(s));
figure();
plot( t_f, 20*log10( abs(spectr_fft) ) )
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')

%% Atomic Norm

% regularizer is off
tau = 1;

y = s;

% convex optimisation
cvx_begin sdp %quiet

variable x_est(N) complex
dual variable x_dual
variable t
variable u(1,N) complex

Z = toeplitz(u);

H = [Z x_est;x_est' t];

minimize( sum_square_abs(y-x_est)/2 + (tau/2)*(real(trace(Z))/N + t) ) %
subject to
H == hermitian_semidefinite(N+1)
x_dual : x_est == y

cvx_end

%% Localise Frequencies

% dual solution
x_hat = x_dual;

% discrete set of Atoms (Fourier)
A = zeros(N, N);
for f = -N/2:N/2-1
    A(:, 1+f+N/2) = exp(1i*2*pi*f*[0:N-1].'/N); 
end
spectr_an = x_hat'*A;


spectr0_fft_dB = 20*log10( abs(spectr_fft) );
spectr_an_dB = 20*log10( abs(spectr_an) );
% normalise FFT spectrum w.r.t. AN's
spectr_fft_dB = spectr0_fft_dB - max(spectr0_fft_dB);

% should be also equivalent to FFT
figure();
hold on
plot( t_f, spectr_an_dB )
plot( t_f, spectr_fft_dB, '--' )
hold off
xlim([t_f(1) t_f(end)])
xlabel('freq (Hz)')
ylabel('spectrum (dB)')
