function [Net] = lm_funcs(Net,x,y,mu,mu_increase_rate,mu_decrease_rate,max_mu,iterations)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%NETWORK WEIGHTS\BIASES STORAGE
IW = Net.IW{1,1};
Ib = Net.b{1,1};
LW = Net.LW{2,1};
Lb = Net.b{2,1};
%%%%%%%%%%%%%%%%%



for i = 1:iterations
    
    while mu <= max_mu && mu > 1e-20
        %PREV-PERFORMANCE COMPUTATION
        Pred = LW*tansig(IW*x + Ib) + Lb; % Y_est
        Prev_Perf = mean((y-Pred).^2); % MSE
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        %PREVIOUS WEIGHTS\BIASES STORAGE
        Prev_IW = IW; % This is for FF, not RBF
        Prev_Ib = Ib;
        Prev_LW = LW;
        Prev_Lb = Lb;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %GRADIENT\HESSIAN COMPUTATION
        [IWJ,IbJ,LWJ,LbJ] = Jacobian_LM(IW,LW,Ib,Lb,x,y);
        [IWUpdate,IbUpdate,LWUpdate,LbUpdate] = UpdatesThroughHessianAndGradient(IWJ,IbJ,LWJ,LbJ,Pred,y,mu);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %WEIGHTS\BIASES UPDATE
        IW = IW + IWUpdate;
        Ib = Ib + IbUpdate;
        LW = LW + LWUpdate';
        Lb = Lb + LbUpdate;
        %%%%%%%%%%%
        
        
        %PERFORMANCE COMPUTATION
        Pred = LW*tansig(IW*x + Ib) + Lb;
        Perf = mean((y-Pred).^2);
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %PERFORMANCE CHECK
        if(Perf >= Prev_Perf)
            IW = Prev_IW;
            Ib = Prev_Ib;
            LW = Prev_LW;
            Lb = Prev_Lb;
            mu = mu*mu_increase_rate;
        else
            mu = mu*mu_decrease_rate;
            break;
        end
        %%%%%%%%%%%%%%%%%%
    end
    
end


%FINAL NET UPDATE
Net.IW{1,1} = IW;
Net.LW{2,1} = LW;
Net.b{1,1} = Ib;
Net.b{2,1} = Lb;
%%%%%%%%%%%


end




function [IWUpdate,IbUpdate,LWUpdate,LbUpdate] = UpdatesThroughHessianAndGradient(IWJ,IbJ,LWJ,LbJ,Pred,y,mu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

s1 = size(IWJ,1);
s2 = size(IWJ,2);
s3 = size(IbJ,1);
s4 = size(LWJ,1);
s5 = size(LbJ,1);
s6 = size(IWJ,3);
Jac = nan(s1*s2 + s3 + s4 + s5,s6);

Jac(1:s1*s2,:) = reshape(IWJ,s1*s2,s6);
Jac(s1*s2+1:s1*s2+s3,:) = IbJ;
Jac(s1*s2+s3+1:s1*s2+s3+s4,:) = LWJ;
Jac(s1*s2+s3+s4+1:s1*s2+s3+s4+s5,:) = LbJ;

H = (Jac*Jac')/s6;
D = mean(Jac.*(Pred - y),2);
Update_Tot = -pinv(H + mu*eye(size(H,1)), min(H(:))/1000)*D;

IWUpdate = reshape(Update_Tot(1:s1*s2),s1,s2);
IbUpdate = Update_Tot(s1*s2+1:s1*s2+s3);
LWUpdate = Update_Tot(s1*s2+s3+1:s1*s2+s3+s4);
LbUpdate = Update_Tot(s1*s2+s3+s4+1:s1*s2+s3+s4+s5);

end





function [IWJ,IbJ,LWJ,LbJ] = Jacobian_LM(IW,LW,Ib,Lb,x,y) %#ok<INUSL>
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%FORWARD PASS
zI = IW*x + Ib;
aI = tansig(zI); % yk in RBF its just vk in output_sim 
%zL = LW*aI + Lb;
%aL = zL;
%%%%%%%%%%%%%%


%BACKPROPAGATION
deltaLW = ones(1,size(y,2));
deltaIW = (1 - aI.^2).*LW'.*deltaLW;
%%%%%%%%%%%%%%%%


%JACOBIAN COMPUTATION
IWJ = nan(size(deltaIW,1),size(x,1),size(x,2));
for i = 1:size(x,2)
    IWJ(:,:,i) = deltaIW(:,i).*x(:,i)';
end

IbJ = deltaIW;

LWJ = aI.*deltaLW;

LbJ = deltaLW;
%%%%%%%%%%%%%%%%%%%%%


end