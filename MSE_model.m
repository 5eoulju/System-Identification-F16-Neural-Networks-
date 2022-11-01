function MSE = MSE_model(X, Y, order)

%{
    Function that get the optimal OLS output and MSE for given dataset and
    maximum order. 
%}

%%% Define parameters
N_states = size(Y, 2); % Number of measurement states
N_meas = size(Y, 1);


MSE = zeros(order, 1, N_states); % Setup MSE matrix for X, Y dataset

%%% Loop through each order
for i = 1:order
    % Regression Matrix Ax 
    Ax = reg_matrix(X, i); 
    
    % Formulate OLS
    theta_OLS = pinv(Ax)*Y; % Measurement set
    Y_est = Ax*theta_OLS;
    
    % Obtain MSE 
    eps = Y-Y_est; % epsilon residual
    
    MSE_input = 1/(N_meas)*(eps'*eps);
    
    MSE(i,1) = MSE_input;
    
end