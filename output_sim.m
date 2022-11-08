function Y_est = output_sim(struct, X)

%{ 
    Function that simulates the RBF neural network using the obtained Wjk weights
    to obtain the estimated output hypothesis.
%} 

%%% Normalization preprocessing
X_norm = mapminmax(X');
X_norm = X_norm'; 

%%% Obtain vj - function from input to hidden
[vj, R] = calc_vj(struct, X_norm);

%%% Use the RBF activation function to get output of the hidden layer
phi_j = exp(-vj);

%%% Use output of the hidden layer to get vk (function of Wjk and output of
%%% hidden layer). In equation form: vk = sum_j(ajk * phi_j)
vk = phi_j * struct.Wjk;

%%% Output layer neuron Y (purelin transfer function)
Y_est_norm = vk;

%%% Reverse normalization for Y_est_norm output
Y_est = mapminmax('reverse', Y_est_norm', struct.postprocess);
Y_est = Y_est'; 

end