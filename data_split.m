function [X_train, X_val, Y_train, Y_val] = data_split(X, Y, train_r, val_r, N_meas);

%%% Method to get indices 
train_idx = fix(train_r * N_meas); % obtain integer number based on ratio
val_idx = fix(val_r * N_meas) + train_idx;

%%% X data value split using indices
X_train = X(1:train_idx, :);
X_val = X(train_idx:val_idx, :);

%%% Y Data value split
Y_train = Y(1:train_idx, :);
Y_val = Y(train_idx:val_idx, :);

end