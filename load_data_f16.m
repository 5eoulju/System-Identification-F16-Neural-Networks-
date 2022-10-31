function [Zk, Uk, Cm] = load_data_f16(file)

%{
    Function to retrieve transposed variables separately

    Outputs:
    > Zk: Measurement Vector
    > Uk: Input Vector
    > Cm: Pitching moment coefficient
%}

load(file, 'Z_k', 'U_k', 'Cm')

%%% Transpose Variables
Zk = Z_k';
Uk = U_k';
Cm = Cm';

end