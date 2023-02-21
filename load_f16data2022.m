% Locate files 
filename = 'Datafile/F16traindata_CMabV_2022';
valname = 'Datafile/F16validationdata_CMab_2022';

% measurement dataset
load(filename, 'Cm', 'Z_k', 'U_k')
% special validation dataset
load(valname, 'Cm_val', 'alpha_val', 'beta_val')

% Output measurement data
alpha_m = Z_k(:,1); % angle of attack
beta_m = Z_k(:,2);  % angle of sideslip
Vtot = Z_k(:,3);    % velocity

% input variables from perfect accelerometer
Au = U_k(:,1); % du/dt data
Av = U_k(:,2); % dv/dt data
Aw = U_k(:,3); % dw/dt data


