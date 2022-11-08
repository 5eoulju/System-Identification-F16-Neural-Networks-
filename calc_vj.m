function [vj, R] = calc_vj(struct, X)

%{
    Function that calculates vj using equation form: vj = sum_i(wij * (xi - cij)^2) 
    > part of (xi - cij)^2 is the squared distance between datapoints and
   given center of each k-mean cluster
    > vj output vector of the input layer
%}

% Squared Distance 
R(:,:,1) = (X(:,1) - struct.centers(:,1)').^2;
R(:,:,2) = (X(:,2) - struct.centers(:,2)').^2;
R(:,:,3) = (X(:,3) - struct.centers(:,3)').^2;

vj = struct.Wij(1,:).*R(:,:,1) + struct.Wij(2,:).*R(:,:,2) + ...
    struct.Wij(3,:).*R(:,:,3);

end
