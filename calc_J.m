function [J, err] = calc_J(net, X, Y)

%{ 
    Function that calculates the Jacobian matrix 
    by propagating the Cost Function dependencies wrt to weights, which in
    turn also updates the weights. 
%}

%%% Parameter setups
N_meas = size(X, 1); % number of measurement points
N_input = size(X, 2); % number of input vars

%%% FeedForward process 
[Y_est, phi_j, vj, R] = output_sim(net, X);

%%% Start backpropagation process 
% 1. Compute dependencies wrt network outputs yk
err_k = Y-Y_est;
dE_dy_k = err_k*-1; 

% 2. Compute dependencies wrt output layer input vk
dy_dvk = 1; % linear activation function

% 3. Compute dependencies wrt hidden layer weights
dvk_dWjk = phi_j; % output of the hidden layer yj
dE_dWjk = dE_dy_k .* dy_dvk .* dvk_dWjk;

% 4. Compute dependencies wrt hidden layer activation function output yj
dvk_dyj = net.Wjk;
% dE_dyj = dE_dy_k .* dy_dvk .* dvk_dyj;

% 5. Compute dependencies wrt hidden layer activation function output yj
% wrt hidden layer inputs vj

dphi_j_dvj = -phi_j; 

% 6. Compute dependencies wrt input weights
dvj_dWij = R.^2; % equal to yi - output of the input layer 

% Complete Partial Derivatives
for i = 1:N_input
    dE_dWij(:,:,i) = dE_dy_k .* dy_dvk .* dvk_dyj' .* dphi_j_dvj .* dvj_dWij(:,:,i);
end

%%% Compute full Jacobian matrix
J = [reshape(dE_dWij, N_meas, net.N_Wij) dE_dWjk];
err = (1/2) * (Y-Y_est).^2; 

end