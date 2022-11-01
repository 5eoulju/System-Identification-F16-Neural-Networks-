% Fig 1: MSE (train, val, test) as function of model order
set(0, 'DefaultFigurePosition', [150 150 720 300])
figure

order = 1:size(MSE, 1);

% subplot(3, 1, 1); hold on
hold on
plot(order, MSE(:, 1, 1), 'b')
% plot(order, MSE(:, 1, 2), 'r')
% plot(order, MSE(:, 1, 3), 'g')
% ylabel('MSE \big[rad$^2$\big]', 'Interpreter', 'Latex')
xlabel('Order [-]', 'Interpreter', 'Latex')
ylabel('MSE [-]', 'Interpreter', 'Latex')
title('MSE vs Model Order: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Measurements', 'Validation'}, 'Location', 'northeast')
legend('boxoff')
grid on