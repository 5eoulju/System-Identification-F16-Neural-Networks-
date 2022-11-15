function [vj, R] = calc_vj(net, X)

%{
    Function that calculates vj using equation form: vj = sum_i(wij * (xi - cij)^2) 
    > part of (xi - cij)^2 is the squared distance between datapoints and
   given center of each k-mean cluster
    > vj output vector of the input layer
%}

N = size(X, 1); % number of measurements
N_states = size(X, 2);
N_neurons = net.N_hidden;
R = zeros(N, N_neurons, N_states); % empty squared distance R 

for i = 1:N_states
    R(:,:,i) = (X(:,i) - net.centers(:,i)').^2;
end

vj = net.Wij(1,:).*R(:,:,1) + net.Wij(2,:).*R(:,:,2) + ...
    net.Wij(3,:).*R(:,:,3);

end
