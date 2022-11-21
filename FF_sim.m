function [Y_est, phi_j, vj] = FF_sim(net, X)

%{ 
    Function that simulates the Feedforward network process using
    normalized input data.
%} 

%%% Normalization preprocessing
if isequal(net.trainAlg{1,1}, 'lg')
    [net, X_norm] = predata_norm(net, X);
elseif isequal(net.trainAlg{1,1}, 'lm')
    X_norm = X; 
end

%%% 1. Obtain vj w/ bias term 
[vj, ~] = calc_vj_FF(net, X_norm);

%%% 2. Use the FF hidden layer activation function (tansig)
phi_j = (exp(vj) - exp(-vj)) ./ (exp(vj) + exp(-vj));

%%% 3. Use output of the hidden layer to get vk 
vk = phi_j * net.Wjk + net.b_jk * net.Wjk;

%%% Output layer neuron Y (purelin transfer function)
Yk = vk;

if isequal(net.trainAlg{1,1}, 'lg')
    %%% Reverse normalization for Y_est_norm output
    Y_est = mapminmax('reverse', Yk', net.postdata);
    Y_est = Y_est'; 
elseif isequal(net.trainAlg{1,1}, 'lm')
    Y_est = Yk;
end

end