% Plotting script for RBF neural network with linear regression.
%
% . - 14.06.2018

% Fig 1: output hypothesis by RBF neural network

% Get hypothesis
Y_est = output_sim_linreg(yRBF_NN, Data_net.X);

% And plot
plot_hypothesis(Data_net.X, Data_net.Y, Y_est, save_fig, 'RBF Network - Linear Regression');