% This file simulates GLV model
% Last modified by Chen Liao on May 17, 2019

function simulatedAbundance = glv_simulation(time, initialAbundance, beta)

ns                      = size(beta, 1);
tol                     = 1e-6;
optionsOde15s           = odeset('RelTol', tol, 'AbsTol', tol * ones(ns,1), 'NonNegative', [1:ns]);
[~,simulatedAbundance]  = ode15s(@generalized_lotka_volterra_model, time, initialAbundance, optionsOde15s, beta);

end

function dXdt = generalized_lotka_volterra_model(t, X, beta)

ns = size(beta,1);
dXdt = zeros(ns,1);

for i=1:ns
    dXdt(i) = beta(i,end) * X(i);
    for j=1:ns
        dXdt(i) = dXdt(i) + X(i) * beta(i,j) * X(j);
    end
end

end