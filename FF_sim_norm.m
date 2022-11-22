function [Yk, phi_j, vj, vi] = FF_sim_norm(net, X)

%{ 
    Function that simulates the Feedforward network process using
    normalized input data.
%} 

%%% Normalization preprocessing
[net, X_norm] = predata_norm(net, X);

%%% 1. Obtain vj w/ bias term 
[vj, vi] = calc_vj_FF(net, X_norm);

%%% 2. Use the FF hidden layer activation function (tansig)
phi_j = (2 ./ (1 + exp(-2*vj))) - 1;

%%% 3. Use output of the hidden layer to get vk 
vk = phi_j * net.Wjk + net.b_jk * net.Wjk;

%%% 4. Output layer neuron Y (purelin transfer function)
Yk = vk;

%%% Reverse normalization for Y_est_norm output
Yk = mapminmax('reverse', Yk', net.postdata);
Yk = Yk'; 

end