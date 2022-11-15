close all; clear; clc;

%% Load Dataset
load('F16reconstructed', 'Z_k1k', 'Cm');
X = Z_k1k'; 
Y = Cm';
N = size(X, 1); 

%% RBF network

% Parameters
eg = 2.98e-5; % sum-squared error goal
sc = 10;    % spread constant

% Network creation
rbf_net = newrb(X', Y', eg, sc, 40);
Y_rbf = rbf_net(X');

%% Plots

figure()
plot3(X(:,1), X(:,2), Y, 'bo', 'MarkerSize', 2)
grid on
hold on
plot3(X(:,1), X(:,2), Y_rbf, 'r-', 'MarkerSize', 2)


