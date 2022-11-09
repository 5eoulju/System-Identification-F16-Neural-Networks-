% Plotting script for RBF neural network with linear regression.
%
% . - 14.06.2018

% Fig 1: output hypothesis by RBF neural network

% Get hypothesis
Y_est = output_sim(yRBF_NN, Data_struct.X);

% And plot
plot_hypothesis(Data_struct.X, Data_struct.Y, Y_est, save_fig, 'RBF Network - Linear Regression');