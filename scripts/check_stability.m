% This function evaluates the local stability of the coexistence steady state
% Last modified by Chen Liao on May 17, 2019

function [isStable, largestEig, relativeAbundance] = check_stability(beta) % beta is a matrix of GLV coefficients

% steady state composition
growthVector        = beta(:, end);                                     % growth rate vector
interactionMatrix   = beta(1:end, 1:end-1);                             % interaction matrix
ssAbundanceIndex    = solve_lin_eq(-growthVector, interactionMatrix);   % steady state solution
relativeAbundance   = ssAbundanceIndex' / sum(ssAbundanceIndex);

% calculate eigenvalues of Jacobian to assess stability
if (min(ssAbundanceIndex) <=0)
    isStable = false; % unseasible steady state
    largestEig = nan;
else
    Jacobian = zeros(size(interactionMatrix));
    for j=1:size(interactionMatrix,1)
        % off diagnol: sum_i b_{i,j} * x_i
        Jacobian(:,j) = ssAbundanceIndex .* interactionMatrix(:, j);
        % diagnol: a_i + sum_j b_{i,j} * x_j + b_{i,i} * x_i
        Jacobian(j,j) = growthVector(j) + interactionMatrix(j,:) * ssAbundanceIndex + ssAbundanceIndex(j) * interactionMatrix(j,j);
    end
    
    largestEig = real(eigs(Jacobian, 1, 'LR')); % real part of largest eigenvalue
    if(largestEig < 0)
        isStable = true;
    else
        isStable = false;
    end
end

end

function xeq = solve_lin_eq(A,B)

[U,S,V] = svd(B);   % Perform SVD on B
s = diag(S);        % vector of singular values
tolerance = max(size(B))*eps(max(s));
p = sum(s>tolerance);
Up = U(:,1:p);
Vp = V(:,1:p);
SpInv = spdiags( 1.0./s(1:p), 0, p, p );
BInv = Vp * SpInv * Up';
xeq = BInv * A;

end