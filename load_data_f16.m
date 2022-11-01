function [Cm, Zk, Uk] = load_data_f16(file)

%{
    Function to retrieve transposed variables separately

    Outputs:
    > Zk: Measurement Vector
    > Uk: Input Vector
    > Cm: Pitching moment coefficient
%}

load(file, 'Cm', 'Z_k', 'U_k')

%%% Transpose Variables
Zk = Z_k';
Uk = U_k';
Cm = Cm';

end