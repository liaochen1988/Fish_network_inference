%   This function estimates gLV parameters using linear regression
%   Last modified by Chen Liao on Dec 3, 2019

function [optBeta,dL,X,C,d] = glv_linreg(time, abundance, lowerbound, upperbound, varargin)

%   abundance has the dimension [# of time] x [# of species]
ny = size(abundance, 1);  %   # of time
ns = size(abundance, 2);  %   # of species

%   abundance values should be all positive
if (sum(abundance(:) <= 0) > 0.5)
    error('abundance has negative value(s).');
end

%   construct dL and X matrix such that dL = beta * X where
%   dL estimates log difference
%   beta stores coefficient matrix
%   X tabulates time-series on species' abundances

%   log-derivatives can be either computed from abundance data or passed in
%   as parameters
if (isempty(varargin))
    % calculate log-derivatives by spline interpolation
    dL  = zeros(ns, ny);   %   [# of species] x [# of time]
    for i=1:ns
        %   using spline to find the correct derivative at each x data
        pp          = spline(time, log(abundance(:,i)));
        pder        = fnder(pp, 1);
        dL(i,:)     = ppval(pder, time);
    end
    
else
    % log-derivatives passed in as parameters
    assert(strcmp(varargin{1}, 'logderiv'));
    dL = varargin{2};
end

X = zeros(ns+1,ny);   %   [# of species + 1] x [# of time]
for i=1:ns
    for j=1:ny
        X(i,j) = abundance(j,i);
    end
end
X(end,:) = 1;

%   construct C/d matrix/vector (such that Cx=d) from dL and X to meet the format requirement of lsqlin
C = [];
for j=1:ns
    C = [C,zeros(size(C,1),size(X',2));zeros(size(X',1),size(C,2)),X'];
end
d = dL';    % d of dimension [# of time] x [# of species]
d = d(:);

%   reformat lower and upper bounds
lb = lowerbound';
lb = lb(:);
ub = upperbound';
ub = ub(:);

%   lsqlin minimizes 0.5 * ||Cx-d||^2
options_lsqlin = optimoptions('lsqlin','Algorithm','interior-point','display','off');
[optBeta, ~, ~, exitflag] = lsqlin(C,d,[],[],[],[],lb,ub,[],options_lsqlin);
if (exitflag <= 0)
    % let us give another try: use trust-region-reflective algorithm
    % instead
    options_lsqlin = optimoptions('lsqlin','Algorithm','trust-region-reflective','display','off');
    [optBeta, ~, ~, exitflag] = lsqlin(C,d,[],[],[],[],lb,ub+1e-6,[],options_lsqlin); 
    if (exitflag <= 0)
        error('lsqlin fails to converge.');
    end
end

optBeta = reshape(optBeta, ns+1, ns);
optBeta = optBeta';

end

