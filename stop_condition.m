        function [stop, early_stop, obj] = stop_condition(obj, e, E, MSE, early_stop)
            % STOP_CRITERIA_LM Checks if criteria for stopping are
            %   fulfilled, where criteria are based on MATLAB's trainlm.
            %
            % Inputs:
            % - obj: object containing the RBF network
            % - e: current epoch number
            % - E: cost function
            % - MSE: array with MSE for train, val, test in each column
            % - early_stop: counter for early stopping
            %
            % Outputs:
            % - stop: whether to stop or not
            % - early_stop: updated counter for early stopping
            % - obj: updated object containing the RBF network
            %
            % . - 06.07.2018
            
            % Compute gradients
            E_grad = gradient(E);
            
            % Increment early stop: if validation error has increased 5
            %   times, stop early
            if e > 1 && MSE(e, 2) > MSE(e-1, 2)
                early_stop = early_stop + 1;
            else
                early_stop = 0;
            end
            
            % Set stop
            stop = 1;
            
            % Case 1: max number of epochs reached
            if e == obj.epochs
                obj.results.stop_criteria = 'max epochs reached';
            % Case 2: cost goal reached
            elseif E(end) < obj.goal
                obj.results.stop_criteria = 'goal reached';
            % Case 3: gradient too low
            elseif e > 2 && abs(E_grad(end)) < obj.min_grad
                obj.results.stop_criteria = 'too low gradient';
            % Case 4: learning rate too high
            elseif obj.mu > obj.mu_max
                obj.results.stop_criteria = 'exceeded mu_max';
            % Case 5: early stopping
            elseif early_stop == 5
                obj.results.stop_criteria = 'early stop';
                e = e - 5;
            % No stop needed
            else
                stop = 0;
            end
            
            % Store results
            obj.results.MSE_train = MSE(:, 1);
            obj.results.MSE_val = MSE(:, 2);
            obj.results.MSE_test = MSE(:, 3);
            obj.results.epoch_optimal = e;
            
        end