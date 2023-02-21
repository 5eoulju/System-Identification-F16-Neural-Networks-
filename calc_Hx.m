function Hx = calc_Hx(x)

%{ 
    Function that calculates the Jabocian of the measurement matrix h
%}

% Setup the states x
u = x(1);
v = x(2);
w = x(3);
C = x(4);

Hx = [-w*(1 + C)/(u^2 + w^2) 0 u*(1 + C)/(u^2 + w^2) atan(w / u);
      (-v*u)/(sqrt(u^2 + w^2)*(u^2 + v^2 + w^2)) sqrt(u^2 + w^2)/(u^2 + v^2 + w^2) -v*w/(sqrt(u^2 + w^2)*(u^2 + v^2 + w^2)) 0;
      u/sqrt(u^2 + v^2 + w^2) v/sqrt(u^2 + v^2 + w^2) w/sqrt(u^2 + v^2 + w^2) 0];

end
