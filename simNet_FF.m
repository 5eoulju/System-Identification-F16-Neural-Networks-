function net = simNet_FF(net, input)

%{
    Function that simulates the FeedForward Neural Network based on the given network
    parameters to approximate any given function. 
    Moreover, it trains the neural network based on two algorithms:
        1. Backpropagation algorithm from Neural Network slides (bp)
        2. Levenberg-Marquardt (lm)
%}

%%% Data Preprocessing
[net, X_train_norm, Y_train_norm] = predata_norm(net, input.X_train, input.Y_train);

%%% Forward prop integration + Backprop training algorithm process
switch net.trainAlg{1,1}
    
    case 'bp' 
        
        %%% Initialization of parameters
        eta = 1; % learning rate 
        Et = zeros(net.epochs, 1); % Cost Function Et 
        MSE = zeros(net.epochs, size(input.X, 2)); % MSE for each input per epoch
        
        %%% Looping through epochs
        for epochs = 1:net.epochs
            % Get estimated Yk with feedforward process + retrieve error
            % term
            [Yk, phi_j, vj, vi] = FF_sim(net, X_train_norm);
            err = MSE_output(Y_train_norm, Yk);
            Et(epochs) = err; % store error term for each epoch
            
            % Apply error backprop training algo using gradient descent 
            grad = backprop(net, X_train_norm, Y_train_norm, Yk, phi_j, vi);
            grad_sum = sum(grad); % sum over all 'q' datapoints

            % Reshape weights into a single row
            wt = reshape([net.Wij' net.Wjk], 1, net.N_weights);
            dw = -eta * grad_sum; 
            w_t1 = wt + dw;
            
            % Updated Weights reshaped to fit into network type
            w_t1_update = reshape(w_t1, net.N_hidden, net.N_input + net.N_output);
            
            net.Wij = w_t1_update(:, 1:net.N_input)';
            net.Wjk = w_t1_update(:, end);
            
            % Get the error output using the updated weights
            Y_train_update = FF_sim(net, X_train_norm);
            err_update = MSE_output(Y_train_norm, Y_train_update);
            
            % condition for the loop to stop
            if err_update < err
                net.Wij = net.Wij;
                net.Wjk = net.Wjk;
                Et(epochs) = err_update;
            else
                %%% Get results for each dataset
                [Y_est_train, ~] = FF_sim_norm(net, input.X_train);
                [Y_est_test, ~] = FF_sim_norm(net, input.X_test);
                [Y_est_val, ~] = FF_sim_norm(net, input.X_val);
                
                %%% MSE
                net.results.MSE(epochs, 1) = MSE_output(input.Y_train, Y_est_train);
                net.results.MSE(epochs, 2) = MSE_output(input.Y_test, Y_est_test);
                net.results.MSE(epochs, 3) = MSE_output(input.Y_val, Y_est_val);
                break;
            end
            
            %%% Get results for each dataset
            [Y_est_train, ~] = FF_sim_norm(net, input.X_train);
            [Y_est_test, ~] = FF_sim_norm(net, input.X_test);
            [Y_est_val, ~] = FF_sim_norm(net, input.X_val);
            
            %%% MSE
            net.results.MSE(epochs, 1) = MSE_output(input.Y_train, Y_est_train);
            net.results.MSE(epochs, 2) = MSE_output(input.Y_test, Y_est_test);
            net.results.MSE(epochs, 3) = MSE_output(input.Y_val, Y_est_val);            
        end 
        
    case 'lm'

        %%% Initialization of parameters
        Et = zeros(net.epochs, 1); % Cost Function Et 
        MSE = zeros(net.epochs, size(input.X, 2)); % MSE for each input per epoch
        
        %%% Get centroids for RBF
        [~, centroid] = kmeans(X_train_norm, net.N_hidden); % get clustered neuron centroid locations
        net.centers = centroid;
        
        %%% Looping through epochs 
        for epochs = 1:net.epochs
            %%% Apply an adaptive learning (mu) algorithm 
            if net.mu <= net.mu_max && net.mu > 1e-20
                %%% Forward propagation to obtain y_est for backpropagation process
                [Y_est_train, phi_j, vj, R] = output_sim(net, X_train_norm);
                
                %%% Compute cost function value Et for each epoch
                Et_epoch = MSE_output(Y_train_norm, Y_est_train);
                Et(epochs) = Et_epoch; % store for each epoch
                
                %%% Obtain weight update: w_t1 = wt-(J'*J+mu*I)^-1*J'*e
                % Compute the Jacobian Matrix J
                [J, err] = calc_J(net, X_train_norm, Y_train_norm, Y_est_train, phi_j, R);
                
                % Compute Hessian matrix 
                H = J'*J;
                
                % Reshape weights into a single row
                wt = reshape([net.Wij' net.Wjk], 1, net.N_weights);
                
                % From weight update w_t1 equation
                lm1 = -pinv(H + net.mu * eye(size(H)));
                lm2 = J'*err;
                dw = lm1 * lm2; % delta w
                
                w_t1 = wt + dw'; % update w_t1 with previous weights wt
                                
                % Updated Weights reshaped to fit into network type
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
                    lm1 = -pinv(H + net.mu * eye(size(H)));
                    lm2 = J'*err;
                    dw = lm1 * lm2; % delta w
                    
                    w_t1 = wt - dw'; % update w_t1
                    
                    w_t1_update = reshape(w_t1, net.N_hidden, net.N_input + net.N_output);
                    net.Wij = w_t1_update(:, 1:net.N_input)';
                    net.Wjk = w_t1_update(:, end);
                    
                    % Get new updated error for new weights given mu
                    Y_train_update = output_sim(net, X_train_norm);
                    err_update = MSE_output(Y_train_norm, Y_train_update);
                end
                           
                net.Wij = w_t1_update(:, 1:net.N_input)';
                net.Wjk = w_t1_update(:, end);
                
                %%% Get results for each dataset
                [Y_est_train, ~] = output_sim_linreg(net, input.X_train);
                [Y_est_test, ~] = output_sim_linreg(net, input.X_test);
                [Y_est_val, ~] = output_sim_linreg(net, input.X_val);
                
                %%% MSE
                net.results.MSE(epochs, 1) = MSE_output(input.Y_train, Y_est_train);
                net.results.MSE(epochs, 2) = MSE_output(input.Y_test, Y_est_test);
                net.results.MSE(epochs, 3) = MSE_output(input.Y_val, Y_est_val);
            else
                net.results.stop_results = 'exceeded mu_max';
                disp("Adaptive Learning exceeded maximum of:")
                net.mu_max;
                disp(" ")
            end
            
            %%% Determine conditions to stop the loop
            [stop_counter, net] = stop_conditions(net, epochs, Et, MSE);
            
            if stop_counter
                break
            end
        end
        
        net.Et = Et;
end
end