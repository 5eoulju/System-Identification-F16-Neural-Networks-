function [X_est_k1k1, Z_k1k_biased, IEKF_count] = func_IEKF(Uk, Zk, dt, sigma_w, sigma_v)

%{
    Function that applies the Iterated Extended Kalman Filter (IEKF)
    > The inputs are stated in the main file
    > The outputs:
        - X_est_k1k1: One-Step-Ahead (K+1,K+1) Optimal State Estimation Vector 
        - Z_k1k_biased: One-Step-Ahead (K+1,K) Measurement Prediction
        Vector
        - IEKF Count: Number of Iterative Measurement Cycles for X_est_k1k1
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IEKF Integration and Simulation Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tstart = 0;
% df = dt;
% tend = 150;
% tspan = tstart:df:tend;
N = size(Uk, 2);
epsilon = 1e-10; % error range for iterative part
apply_iter = 1; % binary switch for iteration
max_iter = 100; % Max. number of iterations possible

%%% Parameters for states
x0 = [Zk(3,1); 0.5; 0.5; 0.5]; % initial state xhat(0|0)
Ex0 = x0; % initial estimate of optimal value Ex0 = xhat(0|0)
states = length(x0);
outputs = size(Zk, 1); 
inputs = size(Uk, 1);
P0 = eye(states) * 0.1; % Initial estimate for the Cov. Matrix of State Prediction error

%%% Process and Measurement Noise Statistics
N_sensor = length(sigma_v); % number of measurements from input sensor
Q = diag(sigma_w.^2); % Process Noise
R = diag(sigma_v.^2); % Sensor Noise
G = eye(states); % System Noise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define arrays and matrices to store results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_est_k1k1 = zeros(states, N);
P_k1k1 = zeros(states, N); % Cov. Matrix State Pred. Error
Z_k1k_biased = zeros(N_sensor, N);
stdx_corr = zeros(states, N);  % Variable to check estimation error in the Iterative Part

% Initial Conditions
x_est_k1_k1 = Ex0; % x(0|0)=E{x_0}
p_k1_k1 = P0;  % P(0|0)=P(0)

IEKF_count = zeros(N, 1);  % Iteration count for each data point

%%% Run IEKF Process
tic; % record current time

ti = 0; 
tf = dt;

% Run the IEKF through all N Measurements
for k = 1:N
    % One-Step Ahead Prediction of x_hat_kk1 
    [t, x_est_k1k] = ode45(@(t, x) calc_F(t, x, Uk(:,k)), [ti tf], x_est_k1_k1); 
    
    % Obtain data from last row
    t = t(end); 
    x_est_k1k = x_est_k1k(end,:)'; 
    
    % One-Step Ahead Prediction Output (z_k1k)
    z_kk1 = calc_h(0, x_est_k1_k1, Uk(:,k)); 
    Z_k1k_biased(:,k) = z_kk1;

    % Step 3: Discretize state transition (Phi_k1k) & input matrix
    % (Gamma_k1k)
    Fx = calc_Fx(0, x_est_k1k, Uk(:,k)); % perturbation of f(x,u,t)
    
    [Phi, Gamma] = c2d(Fx, G, dt); % from cont. to discrete
    
    % Cov. Matrix of State pred. error P_k1k
    P_k1k = Phi*p_k1_k1*Phi' + Gamma*Q*Gamma'; 
    
    
    %%% IEKF Iterative Part: Set Iteration Values & Initial Conditions
    
    % if apply_iter = 1: apply IEKF, else run standard EKF
    if (apply_iter)
        eta2 = x_est_k1k; % init the iteration with state estimation calc before
        err_iter = 2*epsilon;
        N_iter = 0; 
        
        while (err_iter > epsilon)
            
            if (N_iter >= max_iter)
                fprintf('Terminating IEKF: exceeded max iterations (%d)\n', max_iter);
                break
            end
            
            N_iter = N_iter + 1;
            eta1 = eta2;

            % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx = calc_Hx(0, eta1, Uk(:,k)); 
            
%             % Check observability of state
%             if (k == 1 && itts == 1)
%                 rankHF = kf_calcObsRank(Hx, Fx);
%                 if (rankHF < n)
%                     warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
%                 end
%             end

            % The innovation matrix
            Ve = (Hx*P_k1k*Hx' + R);

            % The Kalman gain matrix 
            K = P_k1k * Hx' / Ve;
            
            % Observation State for current loop 
            z_current = calc_h(0, eta1, Uk(:,k)) ; 

            eta2 = x_est_k1k + K * (Zk(:,k) - z_current - Hx*(x_est_k1k - eta1));
            err_iter = norm((eta2 - eta1), inf) / norm(eta1, inf);
        end

        IEKF_count(k) = N_iter;
        x_est_k1_k1 = eta2;

    else
%         % Correction
%         Hx = calc_Hx(0, x_est_k1k, Uk(:,k)); % perturbation of h(x,u,t)
%         % Pz(k+1|k) (covariance matrix of innovation)
%         Ve = (Hx*P_kk_1 * Hx' + R); 
% 
%         % K(k+1) (gain)
%         K = P_kk_1 * Hx' / Ve;
%         % Calculate optimal state x(k+1|k+1) 
%         x_est_k1_k1 = x_kk_1 + K * (Z_k(:,k) - z_kk_1); 

    end    
    
    p_k1_k1 = (eye(states) - K*Hx) * P_k1k * (eye(states) - K*Hx)' + K*R*K';  
    % P_cor = diag(p_k1_k1);
    stdx_corr_x = sqrt(diag(p_k1_k1));

    % Update for next measurement
    ti = tf;
    tf = tf + dt;
    
    % Save data 
    X_est_k1k1(:,k) = x_est_k1_k1;
    P_k1k1(:,k) = diag(p_k1_k1);
    stdx_corr(:,k) = stdx_corr_x;
    
end

end
