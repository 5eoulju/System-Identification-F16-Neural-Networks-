function net = simNet_RBF(net, input)

%{
    Function that simulates the Radial Basis Function Neural Network based on the given network
    parameters to approximate any given function. 
    Moreover, it trains the neural network based on two algorithms:
        1. Linear Regression (lg)
        2. Levenberg-Marquardt (lm)
%}

%%% Data Preprocessing
[net, X_train_norm, Y_train_norm] = predata_norm(net, input.X_train, input.Y_train); 

%%% Forward Prop and Backprop training algorithm process
switch net.trainAlg{1,1}
    
    case 'lg' 
        
        %%% K-means for initial neuron placement
        [~, centroid] = kmeans(X_train_norm, net.N_hidden); 
        net.centers = centroid; 
       
        %%% Input Layer - obtain Vj from input layer
        vj = calc_vj(net, X_train_norm);
        
        %%% Activation function given by phi_j(vj) = a*exp(-vj)
        a = 1;
        phi_j = a*exp(-vj); 
        net.Wjk = pinv(phi_j) * Y_train_norm; % get the hidden-output layer weights
        
        %%% Forward Prop to get estimated for each dataset using obtained Wjk
        [Y_est_train, ~] = output_sim(net, input.X_train);
        [Y_est_test, ~] = output_sim(net, input.X_test);
        [Y_est_val, ~] = output_sim(net, input.X_val);
        
        %%% Obtain the model error using MSE between measured Y set and
        %%% Y_est 
        net.results.MSE_train = MSE_output(input.Y_train, Y_est_train); 
        net.results.MSE_test = MSE_output(input.Y_test, Y_est_test); 
        net.results.MSE_val = MSE_output(input.Y_val, Y_est_val); 
        
    case 'lm'

        %%% Initialization of parameters
        Et = zeros(net.epochs, 1); % Cost Function Et 
        MSE = zeros(net.epochs, size(input.X, 2)); % MSE for each input per epoch
        
        %%% Get centroids for RBF
        [~, centroid] = kmeans(X_train_norm, net.N_hidden); % get clustered neuron centroid locations
        net.centers = centroid;
        
        %%% stop loop conditions
        early_stop = 0;
        
        %%% Looping through epochs 
        
        for epochs = 1:net.epochs
            while net.mu <= net.mu_max && net.mu > 1e-20
                %%% Feedforward to obtain MSE for backpropagation process
                Y_est_train = output_sim(net, X_train_norm);
                
                %%% Compute cost function value Et
                Et_epoch = MSE_output(Y_train_norm, Y_est_train);
                Et(epochs) = Et_epoch; % store for each epoch
                
                %%% Obtain weight update: w_t1 = wt-(J'*J+mu*I)^-1*J'*e
                
                % Compute the Jacobian Matrix J
                [J, err] = calc_J(net, X_train_norm, Y_train_norm);
                
                % Compute Hessian matrix transposed(J)*J
                H = J'*J;
                
                % Reshape weights into an one-liner
                wt = reshape([net.Wij' net.Wjk], 1, net.N_weights);
                
                % From weight update w_t1 equation
                lm1 = -pinv(H + net.mu * eye(size(H)));
                lm2 = J'*err;
                dw = lm1 * lm2; % delta w
                
                w_t1 = wt + dw'; % update w_t1
                
                % w_t1 = wt - ((H + net.mu * eye(size(H))) \ (J' * err))';
                
                % Updated Weights reshaped to fit into net
                w_t1_update = reshape(w_t1, net.N_hidden, net.N_input + net.N_output);
                
                net.Wij = w_t1_update(:, 1:net.N_input)';
                net.Wjk = w_t1_update(:, end);
                
                % Get the error output using the updated weights
                Y_train_update = output_sim(net, X_train_norm);
                err_update = MSE_output(Y_train_norm, Y_train_update);
                
                %%% Apply adaptive learning rate algo based on updated error
                while err_update > err
                    % if updated error is bigger than previous error - increase
                    % learning rate
                    net.mu = net.mu * net.mu_inc;
                    
                    % Weight update given new learning rate
                    % w_t1 = wt - ((H + net.mu * eye(size(H))) \ (J' * err))';
                    lm1 = -pinv(H + net.mu * eye(size(H)));
                    lm2 = J'*err;
                    dw = lm1 * lm2; % delta w
                    
                    w_t1 = wt - dw'; % update w_t1
                    
                    w_t1_update = reshape(w_t1, net.N_hidden, net.N_input + net.N_output);
                    net.Wij = w_t1_update(:, 1:net.N_input)';
                    net.Wjk = w_t1_update(:, end);
                    
                    % Get new updated error for new weights
                    Y_train_update = output_sim(net, X_train_norm);
                    err_update = MSE_output(Y_train_norm, Y_train_update);
                end
                           
                net.Wij = w_t1_update(:, 1:net.N_input)';
                net.Wjk = w_t1_update(:, end);
                
                %%% Get results for each dataset
                [Y_est_train, ~] = output_sim_linreg(net, input.X_train);
                [Y_est_test, ~] = output_sim_linreg(net, input.X_test);
                [Y_est_val, ~] = output_sim_linreg(net, input.X_val);
                
                net.Y_est1 = Y_est_train;
                net.Y_est2 = Y_est_test;
                net.Y_est3 = Y_est_val;
                
                %%% MSE
                MSE(epochs, 1) = MSE_output(input.Y_train, Y_est_train);
                MSE(epochs, 2) = MSE_output(input.Y_test, Y_est_test);
                MSE(epochs, 3) = MSE_output(input.Y_val, Y_est_val);
                
                %%% Determine requirements to stop the loop
                [stop, early_stop, net] = stop_condition(net, epochs, Et, MSE, early_stop);
                
                if stop
                    break
                end
            end 
        end
        
        net.MSE = MSE;
end
end
