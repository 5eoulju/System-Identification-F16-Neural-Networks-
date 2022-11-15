rng(0)
clear
format long

%DEFINE A SIMPLE PROBLEM
x = rand(2,1000)*10;
x = x-5;
y = x(1,:).^2 + x(2,:).^2;
x = x/10;
%%%%%%%%%%%%%%%%%%%%%%%%

%DEFINE TRAINING PARAMETERS
disp(" ")
disp(" ")
Initial_Mu = 0.001;
Incr_Rate = 10;
Decr_Rate = 0.1;
Max_Mu = 1e10;
Epochs = 1000;
Hidden_Neurons = 20;
disp(['Initial_Mu: ',num2str(Initial_Mu)]);
disp(['Incr_Rate: ',num2str(Incr_Rate)]);
disp(['Decr_Rate: ',num2str(Decr_Rate)]);
disp(['Hidden_Neurons: ',num2str(Hidden_Neurons)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DEFINE AND INITIALIZE A NETWORK
net = feedforwardnet(Hidden_Neurons);
net.trainParam.epochs = Epochs;
net.trainParam.mu_dec = Decr_Rate;
net.trainParam.mu_inc = Incr_Rate;
net.trainParam.mu = Initial_Mu;
net.trainParam.mu_max = Max_Mu;
%net.trainParam.showWindow = false;
net.inputs{1,1}.processFcns = {};
net.outputs{1,2}.processFcns = {};
net.trainParam.min_grad = 1e-25;
net.trainParam.max_fail = 50;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net = configure(net,x,y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%USE THAT INITIALIZED NETWORK FOR TRAINING WITH MATLAB'S TRAINLM
disp(" ")
disp(" ")
netMATLAB = net;
tic
netMATLAB = train(netMATLAB,x,y);
disp("Matlab time:")
toc
disp(" ")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%USE THE SAME INITIALIZED NETWORK FOR CUSTOM TRAINING
netCUSTOM = net;
tic
[netCUSTOM] = lm_funcs(netCUSTOM,x,y,Initial_Mu,Incr_Rate,Decr_Rate,Max_Mu,Epochs);
disp("Custom time:")
toc
disp(" ")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%COMPARE RESULTS
Pred_Matlab = netMATLAB(x);
Pred_Custom = netCUSTOM(x);
disp("Absolute difference between Matlab and Custom outputs");
disp(mean(abs(Pred_Matlab - Pred_Custom)));
disp("Matlab MAE")
disp(mean(abs(Pred_Matlab - y)))
disp("Custom MAE")
disp(mean(abs(Pred_Custom - y)))
%%%%%%%%%%%%%%%%