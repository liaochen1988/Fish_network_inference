%   This script applies latent gradient regression to time series of fish population at La Grange pool.
%   Last modified by Chen Liao on Dec 3, 2019

addpath('./scripts'); % tell MATLAB where to find user-defined functions

%%  load time series data

abundanceIndex  = readtable('./data/time_series_LG.csv');           % fishing gear standardized catch-per-unit-effort data
abbrevName      = abundanceIndex.Properties.VariableNames(2:end);   % abbreviated fish common name
time            = abundanceIndex.Time;                              % years when data are available

abundanceIndex  = table2array(abundanceIndex(:,2:end));             % remove header
nt              = length(time);                                     % number of data points in time
ns              = size(abundanceIndex,2);                           % number of fish speices

fprintf('Time series data loaded.\n');

%%  load Sign constraints

%   signConstraints is a table consists of -1 (negative interaction), 0 (neutral interaction), and 1 (positive interaction)
signConstraints = readtable('./data/sign_constraints.xlsx', 'ReadRowNames', true);

lowerbound = zeros(ns, ns+1); % lowerbound for GLV parameters
upperbound = zeros(ns, ns+1); % upperbound for GLV parameters

for row = 1:length(abbrevName)
    rowSpecies = abbrevName(row); % row species
    rowSpeciesIndex = find(contains(signConstraints.Properties.RowNames,rowSpecies)); % index of row species in symbolicConstraints
    
    %   set lower and upper bounds for interaction coefficients
    for col = 1:length(abbrevName)
        colSpecies = abbrevName(col); % column species        
        colSpeciesIndex = find(contains(signConstraints.Properties.VariableNames, colSpecies)); % index of column species in symbolicConstraints
        
        switch signConstraints(rowSpeciesIndex, colSpeciesIndex).Variables
            case -1 % negative interaction
                lowerbound(row, col) = -Inf;
                upperbound(row, col) = 0;
            case 0  % neutral interaction
                lowerbound(row, col) = 0;
                upperbound(row, col) = 0;
            case 1  % positive interaction
                lowerbound(row, col) = 0;
                upperbound(row, col) = Inf;
            otherwise
                error('Invalid sign constraint value.');
        end
    end
    
    %   set lower and upper bounds for population growth rates
    switch signConstraints(rowSpeciesIndex, end).Variables
        case -1 % negative growth rate
            lowerbound(row, end) = -Inf;
            upperbound(row, end) = 0;
        case 1  % positive growth rate
            lowerbound(row, end) = 0;
            upperbound(row, end) = Inf;
        otherwise
            error('Invalid symbolic constraint value.');
    end
end

fprintf('Sign constraints loaded.\n');

%%  data smoothing using empirical mode decomposition
smoothedAbundanceIndex = smoothing(time, abundanceIndex);
fprintf('Data smoothing done.\n');

%%   apply linear regression
[optBetaLR, initialGuessDL] = glv_linreg(time, smoothedAbundanceIndex, lowerbound, upperbound);
fprintf('Solving linear regression done.\n');

%%   apply latent gradient regression
fprintf("Solving latent gradient regression takes about 3 mins. Please be patient.\n");

lambda = [0.000158, 0.007943]; % penalty parameters for interaction matrix and growth vector
optionsLsqnonlin = optimoptions(...
    @lsqnonlin,...
    'Algorithm', 'levenberg-marquardt',...
    'TolFun', 1e-4,...
    'TolX', 1e-4,...
    'MaxFunEvals', Inf,...
    'Display', 'off'...
    );
[optDL,~,~,exitflag] = lsqnonlin(...
    @objective_func,...
    initialGuessDL(:),...
    [],...
    [],...
    optionsLsqnonlin,...
    'logderiv',...
    lambda,...
    time,...
    smoothedAbundanceIndex,...
    lowerbound,...
    upperbound...
    );
if (exitflag <= 0)
    error('lsqnonlin fails to converge.');
end

%   one last run to obtain optimal GLV parameters
optBetaLGR = glv_linreg(...
    time,...
    smoothedAbundanceIndex,...
    lowerbound,...
    upperbound,...
    'logderiv',...
    reshape(optDL, ns, nt)...
    );
fprintf('Solving latent gradient regression done.\n');

%%  figure plot to compare model fitting by different inference algorithms

timeFiner = linspace(time(1), time(end));

%   simulation using GLV inferred from linear regressiion
simulatedAbundanceIndexLR   = glv_simulation (timeFiner, smoothedAbundanceIndex(1,:), optBetaLR);

%   simulation using GLV inferred from latent gradient regression
simulatedAbundanceIndexLGR  = glv_simulation (timeFiner, smoothedAbundanceIndex(1,:), optBetaLGR);

figure();
ylimUB = [3.0, 2.2, 1.6, 1.2, 1.5, 1.0, 1.0, 2.0, 2.0];
for i=1:ns
    subplot(3,3,i);
    hold on;
    plot(time, abundanceIndex(:,i), 'k.', 'MarkerSize',15);
    plot(timeFiner, simulatedAbundanceIndexLR(:,i),'k--','LineWidth',1);
    plot(timeFiner, simulatedAbundanceIndexLGR(:,i),'k-','LineWidth',1);
    box on;
    xlabel('Year');
    ylabel('Abundance index');
    xlim([1992,2016]);
    ylim([0, ylimUB(i)]);
    set(gca,'XTick',[1995,2000,2005,2010,2015]);
    title(char(abbrevName{i}));
    set(gca,'Ticklength',3*get(gca,'Ticklength'));
    legend('Data','LR', 'LGR');
end
