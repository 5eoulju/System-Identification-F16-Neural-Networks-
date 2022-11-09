function [struct, X_norm, varargout] = predata_norm(struct, X, varargin)

%{
    Function to preprocess the data for either input data X and output data
    Y based on normalization techniques (mapminmax function in matlab).
    > varargin collects all input from that point onwards
%}

%%% Pre data settings availability
if isempty(struct.predata) 
    [X_norm, struct.predata] = mapminmax(X'); % put data in range [-1, 1]
    X_norm = X_norm';
else
    X_norm = mapminmax('apply', X', struct.predata); % apply pre data settings if available
    X_norm = X_norm';
end

%%% Post data setting availability
if ~isempty(varargin)
    if isempty(struct.postdata)
        [Y_norm, struct.postdata] = mapminmax(varargin{1}'); 
        varargout{1} = Y_norm';
    else
        Y_norm = mapminmax('apply', varargin{1}', struct.postdata);
        varargout{1} = Y_norm'; 
    end
end
