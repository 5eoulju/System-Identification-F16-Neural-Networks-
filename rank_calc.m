function observ_rank = rank_calc(x, f_state, h_output, x0)

%{
    Function to determine the rank of the observability matrix
    > observability matrix is defined using both the system state (xdot) and
    output equations (z)

    Input arguments are: 
         f_state  - symbolic nonlinear state transition model 
         h_ouput  - symbolic nonlinear observation model
         x  - symbolic state vector
         x0 - numeric initial state vector
%}

%%% Parameters
states = length(x); 
outputs = length(h_output);

%%% Calculate the Rank 
Hx = simplify(jacobian(h_output, x)); % Jacobian of the measurement matrix h
observ_matrix = zeros(outputs*states, states); % init observ matrix 
observ_sym = sym(observ_matrix, 'r'); 
observ_sym(1:outputs, :) = Hx;

observ_full = subs(observ_sym, x, x0); % Get full observability matrix with initial condit.

% Evaluate rank of the observability matrix
observ_rank = double(rank(observ_full));
fprintf('\nRank of Initial Observability matrix is %d\n', observ_rank);

if (observ_rank >= states)
    fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
    return
end

% Get Lie Derivative by multiplying f_state with Hx
Lie_fHx = simplify(Hx * f_state);

% Complete the observability matrix
for i = 2:states
    tic;
    Lie_fHx = jacobian(Lie_fHx, x);
    observ_sym((i-1)*outputs+1:i*outputs, :) = Lie_fHx;
   
    observ_full = subs(observ_sym, x, x0);
    observ_rank = double(rank(observ_full));
    fprintf('\t-> Rank of Observability matrix is %d\n', observ_rank);
    
    if (observ_rank >= states)
        fprintf('Observability matrix is of Full Rank: the state is Observable!\n');
        return
    end
    
    % Next iteration of the Lie Derivative
    Lie_fHx= (Lie_fHx * f_state);
    time = toc;
    fprintf('Loop %d took %2.2f seconds to complete\n', i, toc);

end

fprintf('WARNING: Rank of Observability matrix is %d: the state is NOT OBSERVABLE!\n', observ_rank);


end