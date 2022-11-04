% Fig 1: histogram of residuals to check normal distribution with zero mean
bins = [101 101 301];
figure; hold on
% subplot(3, 1, 1); hold on
histogram(eps_optim(:, 1), bins(1), 'FaceColor', 'b')
xlabel('Residual $C_m$ [-]', 'Interpreter', 'Latex')
ylabel('\# Residuals [-]', 'Interpreter', 'Latex')
title('Model Residual: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
axis([-0.02 0.02 0 500])
grid on

% Fig 2: normalized autocorrelation of residuals and 95% bounds to check
%   correlation
figure; hold on

% subplot(3, 1, 1); hold on
plot(lags, eps_corr(:, 1), 'bo-')
plot(lags, conf_range(1) * ones(1, length(lags)), 'r--')
plot(lags, conf_range(2) * ones(1, length(lags)), 'r--')
xlabel('Lag [-]', 'Interpreter', 'Latex')
ylabel('Normalized Autocorrelation $C_m$ [-]', 'Interpreter', 'Latex')
title('Model Residual Normalized Autocorrelation: $C_m$', 'Interpreter', 'Latex', 'FontSize', 12)
legend({'Normalized autocorrelation', '95\% confidence interval'}, 'Location', 'northeast')
legend('boxoff')
grid on
