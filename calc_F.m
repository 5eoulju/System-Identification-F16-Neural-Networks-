function xdot = calc_F(t, x, u)

%{
    Function that calculates the State Transition Matrix f 
%}

% As mentioned in the assignment - the system states (xdot) is fully
% defined by inputs, so f is trivial
xdot = [u; 0];
end