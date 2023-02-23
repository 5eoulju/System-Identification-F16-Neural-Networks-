function h = calc_h(x, V)

%{
    Function that gets the transformation (observation) matrix H
    for measured data including sensor noise
%}

% Setup states
u = x(1);
v = x(2);
w = x(3);
C = x(4); % C_alpha_up
va = V(1); 
vb = V(2);
vv = V(3);

% true output as function of state variables
atrue = atan2(w, u);
btrue = atan2(v, (sqrt(u.^2+w.^2)));
Vtrue = sqrt(u.^2+v.^2+w.^2);

% Observation equation h(x,u,t) + [va + vb + vv] - Sensor Measured Output
h = [atrue * (1 + C) + va;
    btrue + vb;
    Vtrue + vv];

end
