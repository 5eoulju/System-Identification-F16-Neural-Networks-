
dataname = 'Datafile/F16traindata_CMabV_2022';
valdataname = 'Datafile/F16validationdata_CMab_2022';
% measurement dataset
load(dataname, 'Cm', 'Z_k', 'U_k')
% special validation dataset
load(valdataname, 'Cm_val', 'alpha_val', 'beta_val')

% measurements Z_k = Z(t) + v(t)
alpha_m = Z_k(:,1); % measured angle of attack
beta_m = Z_k(:,2);  % measured angle of sideslip
Vtot = Z_k(:,3);    % measured velocity

% input to Kalman filter
Au = U_k(:,1); % perfect accelerometer du/dt data
Av = U_k(:,2); % perfect accelerometer dv/dt data
Aw = U_k(:,3); % perfect accelerometer dw/dt data


