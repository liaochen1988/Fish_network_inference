%   This file defines the objective function to be minimized
%   Last modified by Chen Liao on Dec 3, 2019

function vvf = objective_func(inputVar, inputVarName, lambda, time, abundance, lowerbound, upperbound)

[nt, ns] = size(abundance);

%   given log-derivatives, run linear regression to obtain optimal GLV parameters
if strcmp(inputVarName, 'logderiv')
    dL = reshape(inputVar, ns, nt);
    optBeta = glv_linreg(time, abundance, lowerbound, upperbound, 'logderiv', dL);
else
    optBeta = reshape(inputVar, ns, ns + 1);
end

%   simulate GLV model using optBeta
simulatedAbundance = glv_simulation(time, abundance(1, :), optBeta);

%   simulation fails
if length(simulatedAbundance(:)) ~= length(abundance(:))
    error('simulation fails.');
else
    vvf                 = simulatedAbundance(:) - abundance(:); % vector-valued function
    interactionMatrix   = optBeta(:,1:ns);
    growthVector        = optBeta(:,end);
    
    % self-interaction coefficients are not panelized
    u1 = zeros(ns*(ns-1),1);
    count=1;
    for i=1:ns
        for j=1:ns
            if (i~=j)
                u1(count) = abs(interactionMatrix(i,j));
                count = count+1;
            end
        end
    end
    u2 = abs(growthVector(:));
    vvf = [vvf; sqrt(lambda(1))*u1; sqrt(lambda(2))*u2];
end

end

