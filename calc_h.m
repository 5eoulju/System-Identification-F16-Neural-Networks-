function Z = calc_h(t, x, u)

%{
    Function that use the observation equation to determine the observation
    matrix h
%}

% Setup states
u = x(1);
v = x(2);
w = x(3);
C = x(4); % C_alpha_up

% This case h is non-trivial in comparison with f
Z = [atan(w / u) * (1 + C);
    atan(v / sqrt(u^2 + w^2));
    sqrt(u^2 + v^2 + w^2)];

% no direct feedback from inputs to output - therefore Du = 0 

end
