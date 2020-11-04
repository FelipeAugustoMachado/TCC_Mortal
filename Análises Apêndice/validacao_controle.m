%==========================================%
% Validating and Simulating Control System %
%==========================================%

clear, clc, close all %Cleaning and closing
warning('off') %Turn off all Matlab warning messages

%% Defining Controler TF:
%Controller with 5% tolerance:
numK = 8.7*[2.82, 1];
denK = [1, 1];
K_TF = tf(numK, denK);

%% Defining Pre-Filter TF:
numF = [25];
denF = [1, 10, 25];

F_TF = tf(numF, denF);

%% Defining Plant Parameters:

% Fixed parameters:
k_t = 70.5e-3; %Torque constant [Nm/A]
r = 0.01; %Motor Reduction
R_phase = 0.343; %Terminal resistance phase-to-phase [Ohm]
R = 3*R_phase; %Parameter ajustment for BLDC model

% Parameter margins:
eta_min = 0.75; %minimum efficiency
eta_max = 0.84; %maximum efficiency

J_min = 0.2198; %minimum inertia value [kg.m^2]
J_max = 0.4082; %maximum inertia value [kg.m^2]


%% Defining simplified parameters and intervals:
% Electric constant (used to compute pole interval):
k_e = 0.0605*k_t; %BLDC motor electric constant

% Gain limits:
K_qft_min = k_t*eta_min/(J_max*R*r);
K_qft_max = k_t*eta_max/(J_min*R*r);

% Pole limits:
P_qft_min = k_t*k_e*eta_min/(J_max*R*r^2);
P_qft_max = k_t*k_e*eta_max/(J_min*R*r^2);

% Defining parameter variation vectors:
K_qft = linspace(K_qft_min, K_qft_max, 10); %10 elements
P_qft = linspace(P_qft_min, P_qft_max, 10); %10 elements

% Defining operation frequency vector:
omega = linspace(0.1, 4.2, 45); %100 elements


%% Validation with Real Plants:

% Defining vectors to contain max values of validations:
% Frequency:
delta_r = []; %array to store all values of delta_r obtained
max_delta_r = [0, 0, 0, 0]; %reference signal tracking - [w, k, p, value]
delta_F = []; %array to store all values of delta_F obtained
max_delta_F = [0, 0, 0, 0]; %pre-filter coupling - [w, k, p, value]
% Time:
s_time = zeros(length(K_qft), length(P_qft)); %array to store all values of settling time obtained
max_s_time = [0, 0, 0]; %step response - [k, p, value]

% Iterating through frequencies and evaluating all real models:

t = linspace(0, 10, 1000); %time vector for frequency analysis

for w = 1:length(omega)
    w_delta_r = 0; %max delta_r value of iteration
    w_delta_F = 0; %max delta_F value of iteration
    
    input = sin(omega(w)*t) + 3; %sinusoidal input with offset
    
    for k = 1:length(K_qft)
        for p = 1:length(P_qft)
            % Defining plant TF:
            numG = [K_qft(k)];
            denG = [1, P_qft(p), 0]; 
            G = tf(numG, denG);
            
            % Calculating signals:
            e_sys = F_TF*feedback(1, K_TF*G); %error signal TF
            
            e = lsim(e_sys, input, t); %error signal
            r = lsim(F_TF, input, t); %closed loop reference signal
            
            e = e.'; %transpose to line
            r = r.'; %transpose to line
            
            % Get only steady state response:
            e = e(300:1000); %from 3 to 10 seconds of simulation
            r = r(300:1000); %from 3 to 10 seconds of simulation
            r1 = input(300:1000); % reference - from 3 to 10 seconds
            
            %% Reference signal tracking validation:
            tolerance_r = abs(e)./abs(r); %calculate ratio for all simulated points
            value_r = max(tolerance_r); %max value for current G and omega
            
            if value_r > max_delta_r(4) %if it is the overall maximum
                max_delta_r(1) = omega(w);
                max_delta_r(2) = K_qft(k);
                max_delta_r(3) = P_qft(p);
                max_delta_r(4) = value_r;
            end
            if value_r > w_delta_r %if it is the max for current omega
                w_delta_r = value_r;
            end
            
            %% Pre-Filter Coupling validation:
            tolerance_F = abs(e)./abs(r1); %calculate ratio for all simulated points
            value_F = max(tolerance_F); %max value for current G and omega
            
            if value_F > max_delta_F(4) %if it is the overall maximum
                max_delta_F(1) = omega(w);
                max_delta_F(2) = K_qft(k);
                max_delta_F(3) = P_qft(p);
                max_delta_F(4) = value_F;
            end
            if value_F > w_delta_F %if it is the max for current omega
                w_delta_F = value_F;
            end
            
            %% Time response validation:
            if w == 1 %don't depends on omega, so run just 1 time
                % Calculating control system step response:
                sys = F_TF*feedback(K_TF*G,1); %control system
                info = stepinfo(sys);
                if info.SettlingTime > max_s_time(3) %update maximum
                    max_s_time(1) = K_qft(k);
                    max_s_time(2) = P_qft(p);
                    max_s_time(3) = info.SettlingTime;
                end
                s_time(k, p) = info.SettlingTime; %append settling time
            end
        end
    end
    % Store max values of this iteration over omega:
    delta_r = [delta_r, w_delta_r]; %append max value to list
    delta_F = [delta_F, w_delta_F]; %append max value to list
end


%% Plot frequency validation:

figure(1)
plot(omega, delta_r, 'b')
title('Max value calculated for $\delta_r$ over each iteration in $\omega$','Interpreter','latex')
xlabel('$\omega$ [rad/s]','Interpreter','latex')
ylabel('max $\delta_r$ value','Interpreter','latex')

figure(2)
plot(omega, delta_F, 'r')
title('Max value calculated for $\delta_F$ over each iteration in $\omega$','Interpreter','latex')
xlabel('$\omega$ [rad/s]','Interpreter','latex')
ylabel('max $\delta_F$ value','Interpreter','latex')

% Printing max values and parameters:
fprintf('Max delta_r value is %f, ', max_delta_r(4))
fprintf('for omega=%f, K_qft=%f and P_qft=%f \n', max_delta_r(1), max_delta_r(2), max_delta_r(3))

fprintf('Max delta_F value is %f, ', max_delta_F(4))
fprintf('for omega=%f, K_qft=%f and P_qft=%f \n', max_delta_F(1), max_delta_F(2), max_delta_F(3))


%% Plot time validation:
fprintf('\nMax settling time: %f ', max_s_time(3))
fprintf('for K_qft=%f and P_qft=%f \n', max_s_time(1), max_s_time(2))

% Calculating plant transfer function for max settling time:
numGmax = [max_s_time(1)];
denGmax = [1, max_s_time(2), 0];
Gmax_TF = tf(numGmax, denGmax);

% Calculating control system:
sys_max = F_TF*feedback(K_TF*Gmax_TF,1);

% Plot step response:
figure(3)
opt = stepDataOptions('StepAmplitude', pi/3);
step(sys_max, opt)
stepinfo(sys_max)

% Plot settling time in parameter surface:
figure(4)
[X, Y] = meshgrid(K_qft, P_qft);
surf(X, Y, s_time)
title('Settling Time $t_s^{2\%}$ value calculated for step response for all $G_R$','Interpreter','latex')
xlabel('$K_{QFT}$','Interpreter','latex')
ylabel('$P_{QFT}$','Interpreter','latex')
zlabel('Settling Time $t_s^{2\%}$','Interpreter','latex')


%% Evaluate control effort, angular velocity and acceleration:

% Control Effort:
sys_u = F_TF*feedback(K_TF,Gmax_TF);

figure(5)
step(sys_u, opt)
title('Control effort u(s) of Step Response')
ylabel('Electric Tension (V)')

% For acceleration and Velocity:
s = tf([1, 0], [1]); %derivate

% Calculate and plot system's angular velocity:

sys_veloc = sys_max*s;

figure(6)
step(sys_veloc, opt)
title('Angular Velocity of Step Response')
ylabel('Amplitude [rad/s]')

% Calculate and plot system's angular acceleration:

sys_accel = sys_veloc*s;

%Filter aceeleration - cutoff in 5Hz:
n = 2; %filter order
fc = 5; %desired cutoff frequency (Hz)
fs = 63; %sampling frequency
Wn = fc/(fs/2); %cutoff frequency (rad)
[num_filter, den_filter] = butter(n,Wn); %generate filter values

figure(7)
[alpha,t] = step(sys_accel, opt);
plot(t, alpha, 'b')
hold on
alpha_filt = filtfilt(num_filter, den_filter, alpha);
plot(t, alpha_filt, 'red')
hold off
title('Angular Acceleration of Step Response')
ylabel('Amplitude [rad/$s^2$]', 'Interpreter', 'latex')
legend('Not filtered', 'Filtered')

fprintf('\nMax angular acceleration: %f ', max(alpha_filt))