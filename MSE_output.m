function MSE = MSE_output(Ym, Yest)

%{
    Function to calculate the Mean-Squared Error (MSE) between estimated Y
    and measurement data Y
%}
N_meas = size(Ym, 1);
epsilon = Ym - Yest;
MSE = (epsilon' * epsilon) / (2*N_meas);
MSE = diag(MSE); % get MSE from the matrix diagonals MSE

end
