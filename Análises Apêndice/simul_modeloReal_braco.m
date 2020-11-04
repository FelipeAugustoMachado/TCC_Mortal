%=============================================%
% Simulating Frequency Response - Exoskeleton %
%=============================================%

clear, clc, close all %Cleaning and closing

%% Model Data:

n = 1000; %Number of elements

%Exoskeleton:

J = 5.2e-4; %Inertia defined with respect to motor rotation axis [kg*m^2]

dk = 68.9e9; %Modulus of elasticity for aluminum 6061 [Pa] 
dJ = J/n;

%BLDC motor:

r = 0.01; %Motor Reduction
eta = 0.84; %Max. efficiency
k_t = 70.5e-3; %Torque constant [Nm/A]
k_e = 0.0605*k_t; %BLDC motor electric constant

R_phase = 0.343; %Terminal resistance phase-to-phase [Ohm]
R = 3*R_phase; %Parameter ajustment for BLDC model


%
%% Defining frequency range:

nw = 1000; %number of frequency values
w = logspace(0, 6, nw); %building log scale frequency values

%
%% Calculating multiplicative error:

%Building A and B matrices of linear system A*theta = B
V = 1; %electric tension [V]
B = zeros(n,1);
B(1) = eta*k_t*V/(R*r);

A = zeros(n, n);

lm = []; %multiplicative error vector

%Iterating trough w vector:
for i=1:length(w)
    jw = j*w(i);
    jw2 = jw*jw;
    
    %Calculating G for current frequency
    G = (r/k_e)/(jw*(J*R*(r^2)*jw/(k_t*k_e*eta) + 1));
    
    %Building A matrix:
    alpha = jw2*dJ - jw*eta*k_t*k_e/(R*r^2) + 2*dk;
    beta = jw2*dJ + 2*dk;
    gamma = jw2*dJ + dk;
    %first line:
    A(1,1) = alpha;
    A(1, 2) = -dk;
    %defining lines 2 to n-1:
    for j=2:n-1
        A(j, j-1) = -dk;
        A(j, j) = beta;
        A(j, j+1) = -dk;
    end
    %last line:
    A(n, n-1) = -dk;
    A(n, n) = gamma;
    
    %Solving linear system:
    theta = linsolve(A, B); %solving
    Gr = theta(n); %obtaining real transfer function for current frequency
    lm_now = abs((Gr-G)/G); %current multiplicative error
    lm = [lm lm_now]; %append current lm to lm vector
end

%
%% Plot multiplicative error graphic:

lm_dB = 20*log10(lm);
semilogx(w,lm_dB,'LineWidth', 1.5)
grid
title('Multiplicative Error for Deformable Exoskeleton Model')
xlabel('Angular frequency (rad/s)')
ylabel('l_{m} (dB)')