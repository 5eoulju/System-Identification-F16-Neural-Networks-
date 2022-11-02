function chart_MSE(MSE_meas, MSE_val, order)

x = 1:order; % x-axis range

figure()
hold on
plot(x, MSE_meas, 'b')
plot(x, MSE_val, 'r')
xlabel('Order [-]', 'Interpreter', 'Latex')
ylabel('MSE [-]', 'Interpreter', 'Latex')
title('MSE vs Model Order: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Measurements', 'Validation'}, 'Location', 'northeast')
legend('boxoff')
grid on

end