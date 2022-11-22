%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AE4320 System Identification of Aerospace Vehicles 21/22
% Assignment: Neural Networks
% 
% Part 4 Code: FeedForward Neural Network
% Date: 15 NOV 2022
% Creator: J. Huang | 4159772
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc
plot = 0; save_fig = 0;

%% Load Dataset
%%% Split data into training, validation and testing data
load('Datafile/F16reconstructed', 'Z_k1k', 'Cm') % obtain reconstructed data from part 2
X = Z_k1k'; 
Y = Cm';
N = size(X, 1); % number of measurements

%%% Ratio split of data
train_r = 0.6; % Training ratio
test_r = 0.25;
val_r = 0.15; 
data_ordering = randperm(N); % randomly ordering the data to avoid bias

X = X(data_ordering, :); % apply new orderly data X
Y = Y(data_ordering, :); % and to Y

%%% Get randomly ordered splitted data for both X & Y
[X_train, X_test, X_val, Y_train, Y_test, Y_val] = update_data(X, Y, train_r, test_r, val_r, N);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of FeedForward Neural Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 1. Create FF Neural Network Struct. Template 

%%% Object input parameters field (see appendix D of assignment)
% Fixed fields
FF_net.name = 'feedforward';
FF_net.trainFunc = {'tansig', 'purelin'}; % Activation function per layer
FF_net.trainAlg = {'bp'}; % training algorithm 

% Layer parameters 
FF_net.N_input = 3; % alpha, beta, V
FF_net.N_hidden = 50; % number of neurons
FF_net.N_hidden_layers = 1; % number of hidden layers
FF_net.N_output = 1; % Cm

% Initialization of weights parameters
FF_net.b_ij = ones(FF_net.N_hidden, FF_net.N_input); % input bias weights
FF_net.b_jk = ones(FF_net.N_output, 1); % output bias weights
FF_net.N_Wij = FF_net.N_input * (FF_net.N_hidden+1); % add a weight term for bias
FF_net.N_Wjk = FF_net.N_hidden;
FF_net.N_weights = FF_net.N_hidden * (FF_net.N_input + FF_net.N_output); % total weights
FF_net.Wij = randn(FF_net.N_input, FF_net.N_hidden); % Weights ij from input to hidden assuming 1 hidden layer
FF_net.Wjk = randn(FF_net.N_hidden, FF_net.N_output); % Weights jk from hidden to output 
input_range = [-ones(FF_net.N_input, 1), ones(FF_net.N_input, 1)]; % bound to input space

% Other parameters 
FF_net.epochs = 100; 
FF_net.goal = 1e-6;  % Desired performance reached 
FF_net.min_grad = 1e-10; % training stops when gradient below value
FF_net.mu = 0.001; % Learning rate parameters
FF_net.mu_dec = 0.1;
FF_net.mu_inc = 10;
FF_net.mu_max = 1e10;

%%% Adding empty fields for data processing
FF_net.predata = []; % prep data before putting into network
FF_net.postdata = []; % prep output data to use for analytics

%%% Put data into struct type
Data_net = struct('X', X, 'X_train', X_train, 'X_test', X_test, 'X_val', X_val, ...
    'Y', Y, 'Y_train', Y_train, 'Y_test', Y_test, 'Y_val', Y_val); % IO data points in struct type
%%
%%% 2. IO mapping - Simulating network output 
yFF_NN = simNet_FF(FF_net, Data_net);

%%% 3. Results
plot_FF

%% Backtesting

%%% Data Preprocessing
[FF_net, X_train_norm, Y_train_norm] = predata_norm(FF_net, Data_net.X_train, Data_net.Y_train);

%%% Initialization of parameters
eta = 1; % learning rate
Et = zeros(FF_net.epochs, 1); % Cost Function Et
MSE = zeros(FF_net.epochs, size(Data_net.X, 2)); % MSE for each input per epoch

%%% Looping through epochs
for epochs = 1:FF_net.epochs
    % Get estimated Yk with feedforward process + retrieve error
    % term
    % [Yk, phi_j, vj, vi] = FF_sim(FF_net, X_train_norm);
    
    %%%%%%%%%%%%%%%%%%%%%%%
    
    %%% 1. Obtain vj w/ bias term
    % [vj, vi] = calc_vj_FF(FF_net, X_train_norm);
    
    %%%%%%%%%%%%%%%%%%%%%%%
    
    N = size(X_train_norm, 1); % number of measurements
    N_states = size(X_train_norm, 2);
    N_neurons = FF_net.N_hidden;
    vi = zeros(N, N_neurons, N_states); % empty vi for each dataset of xi
    
    for i = 1:N_states
        vi(:,:,i) = (X_train_norm(:,i) + FF_net.b_ij(:,i)'); % b_ij has 50 neurons giving vi(:, N_neurons, N_states)
    end
    
    vj = FF_net.Wij(1,:).*vi(:,:,1) + FF_net.Wij(2,:).*vi(:,:,2) + ...
        FF_net.Wij(3,:).*vi(:,:,3); % have to add a Weight term for Bias

    
    %%%%%%%%%%%%%%%%%%%%%%%

    %%% 2. Use the FF hidden layer activation function (tansig)
    phi_j = (2 ./ (1 + exp(-2*vj))) - 1;
    
    %%% 3. Use output of the hidden layer to get vk
    vk = phi_j * FF_net.Wjk + FF_net.b_jk * FF_net.Wjk;
    
    %%% 4. Output layer neuron Y (purelin transfer function)
    Yk = vk;
    
    %%%%%%%%%%%%%%%%%%%%%%%
    
    err = MSE_output(Y_train_norm, Yk);
    Et(epochs) = err; % store error term for each epoch
    
    % Apply error backprop training algo using gradient descent
    grad = backprop(FF_net, X_train_norm, Y_train_norm, Yk, phi_j, vi);
    grad_sum = sum(grad); % sum over all 'q' datapoints
    
    % Reshape weights into a single row
    wt = reshape([FF_net.Wij' FF_net.Wjk], 1, FF_net.N_weights);
    dw = -eta * grad_sum;
    w_t1 = wt + dw;
    
    % Updated Weights reshaped to fit into network type
    w_t1_update = reshape(w_t1, FF_net.N_hidden, FF_net.N_input + FF_net.N_output);
    
    FF_net.Wij = w_t1_update(:, 1:FF_net.N_input)';
    FF_net.Wjk = w_t1_update(:, end);
    
    % Get the error output using the updated weights
    Y_train_update = FF_sim(FF_net, X_train_norm);
    err_update = MSE_output(Y_train_norm, Y_train_update);
    
    % condition for the loop to stop
    if err_update < err
        FF_net.Wij = FF_net.Wij;
        FF_net.Wjk = FF_net.Wjk;
        Et(epochs) = err_update;
    else
        break;
    end
    
    %%% Get results for each dataset
    [Y_est_train, ~] = FF_sim_norm(FF_net, Data_net.X_train);
    [Y_est_test, ~] = FF_sim_norm(FF_net, Data_net.X_test);
    [Y_est_val, ~] = FF_sim_norm(FF_net, Data_net.X_val);
    
    %%% MSE
    FF_net.results.MSE(epochs, 1) = MSE_output(Data_net.Y_train, Y_est_train);
    FF_net.results.MSE(epochs, 2) = MSE_output(Data_net.Y_test, Y_est_test);
    FF_net.results.MSE(epochs, 3) = MSE_output(Data_net.Y_val, Y_est_val);
end
