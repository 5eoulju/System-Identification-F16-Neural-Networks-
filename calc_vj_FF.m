function [vj, vi] = calc_vj_FF(net, X)

%{
    Function that calculates vj using equation form: vj = sum(Wij * phi_i(vi) + Wij*bi)
    Where vi = xi + bi 
    > vj output vector of the input layer
%}

N = size(X, 1); % number of measurements
N_states = size(X, 2);
N_neurons = net.N_hidden;
vi = zeros(N, N_neurons, N_states); % empty vi for each dataset of xi 

for i = 1:N_states
    vi(:,:,i) = (X(:,i) + net.b_ij(:,i)'); % b_ij has 50 neurons giving vi(:, N_neurons, N_states)
end

vj = net.Wij(1,:).*vi(:,:,1) + net.Wij(2,:).*vi(:,:,2) + ...
    net.Wij(3,:).*vi(:,:,3); % have to add a Weight term for Bias 
