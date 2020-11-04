%============================================%
% Simulating Frequency Response - BLDC Motor %
%============================================%

clear, clc, close all %Cleaning and closing

%% Model Data:

%Exoskeleton:

J = 5.2e-4; %Inertia defined with respect to motor rotation axis [kg*m^2]

%BLDC motor:

r = 0.01; %Motor Reduction
eta = 0.84; %Max. efficiency
k_t = 70.5e-3; %Torque constant [Nm/A]
k_e = 0.0605*k_t; %BLDC motor electric constant

L = 0.264e-3; %Terminal inductance [H]
R_phase = 0.343; %Terminal resistance phase-to-phase [Ohm]
R = 3*R_phase; %Parameter ajustment for BLDC model

tau_m = (R*J*r^2)/(k_t*k_e*eta); %Mechanical time constant [s]
tau_e = L/R; %Electrical time constant [s]

%
%% Defining frequency range:

nw = 5000; %number of frequency values
w = logspace(-3, 6, nw); %building log scale frequency values

%
%% Calculating multiplicative error:

lm = []; %multiplicative error vector

%Iterating trough w vector:
for i=1:length(w)
    jw = j*w(i);
    jw2 = jw*jw;
    
    %Calculating G and Gr for current frequency
    G = (r/k_e)/(jw2*tau_m + jw); %nominal
    Gr = (r/k_e)/(jw2*tau_m*tau_e + jw*tau_m + 1); %real
    lm_now = abs((Gr-G)/G); %current multiplicative error
    lm = [lm lm_now]; %append current lm to lm vector
end

%
%% Plot multiplicative error graphic:

lm_dB = 20*log10(lm);
semilogx(w,lm_dB,'LineWidth', 1.5)
grid
title('Multiplicative Error for BLDC Model with Inductance')
xlabel('Angular frequency (rad/s)')
ylabel('l_{m} (dB)')