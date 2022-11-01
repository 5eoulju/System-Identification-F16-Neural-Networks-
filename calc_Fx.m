function Fx = calc_Fx(t, x, u)

%{ 
    Function that calculates the Jacobian of the state transition matrix f
%}

% since f is trivial -> partial derivative basically outputs zero
Fx = zeros(4);
end