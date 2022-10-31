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
states = 4; 
% inputs = 3; 
% u0 = [5; 5; 5];
x0 = [100; 10; 10; 1];

%%% Define System State and Output Equations as described as above
x = [u; v; w; C_alpha_up]; % state vector
f_state = [udot; vdot; wdot; 0]; % state transition matrix f 

h_output = [atan(w/u) * (1 + C_alpha_up); % Measurement matrix h
    atan(v/sqrt(u^2 + w^2));
    sqrt(u^2 + v^2 + w^2)];

rank = rank_calc(x, f_state, h_output, x0); % check rank using rank_calc func

if rank >= states
    fprintf('\nSystem is observable (Observability Matrix = Full Rank)\n');
else
    fprintf('\nSystem is not observable (Observability Matrix = Not Full Rank)\n');
end


end