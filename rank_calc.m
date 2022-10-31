function observ_rank = rank_calc(x, f_state, h_output, x0)

%{
    Function to determine the rank of the observability matrix
    > observability matrix is defined using both the system state (xdot) and
    output equations (z)
%}

%%% Parameters
states = length(x); 
outputs = length(h_output);
observ_matrix = zeros(outputs*states, states); % init observ matrix 
observ_sym = sym(observ_matrix, 'r'); % get rationale form to avoid decimal outcomes

%%% Calculate the Rank 
Hx = simplify(jacobian(h_output, x)); % Jacobian of the measurement matrix h
observ_sym(1:length(h_output), :) = Hx; % Substitute Hx into observ matrix
observ_full = subs(observ_sym, x, x0); % Get full observability matrix with initial condit.

% Evaluate rank of the observability matrix
observ_rank = double(rank(observ_full));

if observ_rank >= states 
    return
end

% Get Lie Derivative by multiplying f_state with Hx
Lie_fHx = simplify(Hx * f_state);

% Complete the observability matrix
for i = 2:states
    Lie_fHx = jacobian(Lie_fHx, x);
    observ_sym((i-1)*outputs+1:i*outputs, :) = Lie_fHx;
   
    observ_full = subs(observ_sym, x, x0);
    observ_rank = double(rank(observ_full));
    
    if observ_rank >= states
        return
    end
    
    % Next iteration of the Lie Derivative
    Lie_fHx= (Lie_fHx * f_state);
end

end