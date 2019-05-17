%   This file defines the objective function to be minimized
%   Last modified by Chen Liao on May 17, 2019

function vvf = objective_func(inputVar, inputVarName, lambda, time, abundance, lowerbound, upperbound, trainSet)

[nt, ns] = size(abundance);

%   given log-derivatives, run linear regression to obtain optimal GLV parameters
if strcmp(inputVarName, 'logderiv')
    dL = reshape(inputVar, ns, nt);
    optBeta = glv_pe_linreg(time, abundance, lowerbound, upperbound, trainSet, 'logderiv', dL);
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
    vvf                 = vvf(trainSet);
    interactionMatrix   = optBeta(:,1:ns);
    growthVector        = optBeta(:,end);
    vvf                 = [vvf; sqrt(lambda(1))*interactionMatrix(:); sqrt(lambda(2))*growthVector(:)];
end

end

