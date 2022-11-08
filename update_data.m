function [X_train, X_test, X_val, Y_train, Y_test, Y_val] = update_data(X, Y, train_r, test_r, val_r, N_meas)

%%% Method to get indices 
train_idx = fix(train_r * N_meas); % obtain integer number based on ratio
test_idx = fix(test_r * N_meas) + train_idx;
val_idx = fix(val_r * N_meas) + test_idx;

%%% X data value split using indices
X_train = X(1:train_idx, :);
X_test = X(train_idx:test_idx, :);
X_val = X(test_idx:val_idx, :);

%%% Y Data value split
Y_train = Y(1:train_idx, :);
Y_test = Y(train_idx:test_idx, :);
Y_val = Y(test_idx:val_idx, :);

end

