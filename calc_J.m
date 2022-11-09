function [J, err] = calc_J(struct, X, Y)

%{ 
    Function that calculates the Jacobian matrix of a structure type input
%}

%%% FeedForward process 
[Y_est, phi_j, vj, R] = output_sim(struct, X);

%%% Gradient error vector over hidden layer weights wjk
N_meas = size(X, 1); % number of measurement points
N_input = size(X, 2); % number of input vars

%%% Start backpropagation process 
% 1. Compute dependencies wrt network outputs
err_k = Y-Y_est;
dE_dy_k = err_k*-1;

% 2. Compute dependencies wrt output layer input vk
dy_dvk = 1; % linear activation function

% 3. Compute dependencies wrt hidden layer weights
dvk_dw_jk = phi_j;
dE_dw_jk = dE_dy_k .* dy_dvk .* dvk_dw_jk;

% 4. Compute dependencies wrt hidden layer activation function output yj
dvk_dyj = struct.Wjk;

%%% Gradient error vector over input layer weights Wij
% 5. Compute dependencies of hidden layer activation function outputs yj
% wrt hidden layer inputs vj
dphi_j_dvj = -phi_j;

% 6. Compute dependencies wrt input weights
dvj_dwij = R.^2;

% 7. Complete Partial Derivatives
for i = 1:N_input
    dE_dwij(:,:,i) = dE_dy_k .* dy_dvk .* dvk_dyj' .* dphi_j_dvj .* dvj_dwij(:,:,i);
end

%%% Compute full Jacobian matrix
J = [reshape(dE_dwij, N_meas, struct.N_input*struct.N_hidden) dE_dw_jk];
err = (Y-Y_est).^2 / 2; 

end