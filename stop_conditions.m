function [stop_counter, net] = stop_conditions(net, epochs, Et, MSE)

%{
    Function that checks whether certain criteria are met in order to stop
    the algorithm from running much longer. 
    Implemented Criteria:
        1. Desired performance reached (goal)
        2. Max epoch reached
        3. Gradient Descent has reached its min_grad
%}

stop_counter = 1; % default
grad = gradient(Et); % Compute gradients of Et per epoch

%%% Form conditional statement for cases

% Case 1: Desired Performance reached (goal)
if Et(epochs) < net.goal
    net.results.stop = 'Desired Performance Reached';
% Case 2: Max Epochs Reached
elseif epochs == net.epochs
    net.results.stop = 'Maximum Epochs Reached';
% Case 3: Gradient Descent reached min_grad
elseif epochs > 2 && abs(grad(epochs)) < net.min_grad
    net.results.stop = 'Gradient has reached its min_grad';
else
    stop_counter = 0;
end

% Put results into output network
net.results.MSE_train = MSE(:, 1);
net.results.MSE_val = MSE(:, 2);
net.results.MSE_test = MSE(:, 3);
net.results.opt_epochs = epochs;

end