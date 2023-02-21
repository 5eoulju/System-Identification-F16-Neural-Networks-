function observ_check()

%{
    Function to check whether the system states are observable in order for the KF to
    converge properly. 
%}

%%% Symbols use for this func
syms('u', 'v', 'w','C_alpha_up', 'udot', 'vdot', 'wdot')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 2.1 - General overview of f16 state and output system eqs
%   1) System State Equation: xdot(t) = f(x(t), u(t), t)
%         > State Vector: x(t) = [u v w C_alpha_up]
%         > Input Vector: u(t) = [udot, vdot, wdot]
%     
%   2) Measurement (Output) Equation: z(t) = h(x(t), u(t), t)
%         > Output vector: z(t) = [alpha_m beta_m V_m]
%         > Additional + [v_alpha v_beta v_V] as white noise 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Parameters & Initial Conditions
x0 = [100; 10; 10; 1];
x = [u; v; w; C_alpha_up]; % state vector
f_state = [udot; vdot; wdot; 0]; % state transition matrix f
sens_noise = [u; v; w];
h_output = calc_h(x, sens_noise);

rank_calc(x, f_state, h_output, x0); % check rank using rank_calc func

end