function [eps_corr, conf_range, lags] = model_err_val(eps)

%%%  Compute autocorrelation of residual of eps with its mean
[eps_corr, lags] = xcorr(eps - mean(eps, 1));

eps_corr = eps_corr ./ max(eps_corr, 1);

conf_range = sqrt(1 / size(eps_corr, 1)) * [-3; 3];% set confidence range of 95% 

end
