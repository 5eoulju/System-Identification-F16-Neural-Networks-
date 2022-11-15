function [Y_est, phi_j, vj, R] = output_sim_linreg(net, X)

%{ 
    Function that simulates the RBF neural network using the obtained Wjk weights
    to obtain the estimated output hypothesis.
%} 

%%% Normalization preprocessing
[net, X_norm] = predata_norm(net, X);

%%% Obtain vj - function from input to hidden
[vj, R] = calc_vj(net, X_norm);

%%% Use the RBF activation function to get output of the hidden layer
phi_j = exp(-vj);

%%% Use output of the hidden layer to get vk (function of Wjk and output of
%%% hidden layer). In equation form: vk = sum_j(ajk * phi_j)
vk = phi_j * net.Wjk;

%%% Output layer neuron Y (purelin transfer function)
Y_est_norm = vk;

%%% Reverse normalization for Y_est_norm output
Y_est = mapminmax('reverse', Y_est_norm', net.postdata);
Y_est = Y_est'; 

end