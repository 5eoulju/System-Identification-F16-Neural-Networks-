function MSE = MSE_model(X, Y, order)

%{
    Function that get the MSE for given maximum order 
%}

%%% Define parameters
N_meas = size(Y, 1); % Measurement number
MSE = zeros(order, 1); % Setup MSE matrix for X, Y dataset

%%% Loop through the order and get and store MSE for each loop
for i = 1:order
    % Use OLS estimator script to get Y_est
    Ax = reg_matrix(X, i); 
    theta_OLS = pinv(Ax)*Y; 
    Y_est = Ax*theta_OLS;
    
    % Obtain MSE using Y_est and measurement Y data
    eps = Y-Y_est; % epsilon residual
    MSE_input = 1/N_meas*(eps'*eps); % MSE eq. in matrix notation
    MSE(i,1) = MSE_input; % store data
end