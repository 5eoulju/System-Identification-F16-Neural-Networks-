function [polynomial_order, err] = optimal_order(X_train, Y_train, X_val, Y_val, ols_type)

%{
    Find most optimal order for data approximation
%} 
polynomial_order = 0;
err_current = inf;
err_new = 1e30;
err = [];

% figure()

while err_new < err_current
    err_current = err_new;
    [Ax_train, theta] = reg_matrix(polynomial_order, X_train, Y_train, ols_type);
    [Ax_val,~] = reg_matrix(polynomial_order, X_val, Y_val, ols_type);
    
    % Cm output 
    y_output = Ax_val*theta;
    residual = Y_val - y_output;
    err_new = sum(residual.^2) / size(residual, 1);
    
    err = [err; err_new];
    
%     cla
%     plot3(X_train(:,1), X_train(:,2), y_output, '.k');
%     
%     hold on
%     trisurf(TRIeval, a_true_2, b_true_2, Cm, 'EdgeColor', 'None');
%     
%     refreshdata
%     grid on;
%     title(strcat('sumorder, MSE=', num2str(err_new)));
%     legend('Linear Regression Model', 'full dataset');
%     pause(0.1)
    
    polynomial_order = polynomial_order + 1;
end

polynomial_order = polynomial_order - 2;


end

