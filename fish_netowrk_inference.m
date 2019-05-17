%   This script applies latent variable regression to time series of fish population at La Grange pool.
%   Last modified by Chen Liao on May 16, 2019

addpath('./scripts'); % tell MATLAB where to find user-defined functions

%%  load time series data

abundanceIndex  = readtable('./data/time_series_LG.csv');           % fishing gear standardized catch-per-unit-effort data
abbrevName      = abundanceIndex.Properties.VariableNames(2:end);   % abbreviated fish common name
time            = abundanceIndex.Time;                              % years when data are available

abundanceIndex  = table2array(abundanceIndex(:,2:end));             % remove header
nt              = length(time);                                     % number of data points in time
ns              = size(abundanceIndex,2);                           % number of fish speices

fprintf('Time series data loaded.\n');

%%  load symbolic constraints

%   symbolicConstraints is a table consists of -1 (negative interaction), 0 (neutral interaction), and 1 (positive interaction)
symbolicConstraints = readtable('./data/symbolic_constraints.xlsx', 'ReadRowNames', true);

lowerbound = zeros(ns, ns+1); % lowerbound for GLV parameters
upperbound = zeros(ns, ns+1); % upperbound for GLV parameters

for row = 1:length(abbrevName)
    rowSpecies = abbrevName(row); % row species
    rowSpeciesIndex = find(contains(symbolicConstraints.Properties.RowNames,rowSpecies)); % index of row species in symbolicConstraints
    
    %   set lower and upper bounds for interaction coefficients
    for col = 1:length(abbrevName)
        colSpecies = abbrevName(col); % column species        
        colSpeciesIndex = find(contains(symbolicConstraints.Properties.VariableNames, colSpecies)); % index of column species in symbolicConstraints
        
        switch symbolicConstraints(rowSpeciesIndex, colSpeciesIndex).Variables
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
                error('Invalid symbolic constraint value.');
        end
    end
    
    %   set lower and upper bounds for population growth rates
    switch symbolicConstraints(rowSpeciesIndex, end).Variables
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

fprintf('Symbolic constraints loaded.\n');

%%  data smoothing using empirical mode decomposition
testSet = zeros(size(abundanceIndex(:))); %   use all data as training dataset
trainSet = ~testSet;
smoothedAbundanceIndex = smoothing(time, abundanceIndex, reshape(trainSet, nt, ns));

fprintf('Data smoothing done.\n');

%%  apply latent variable regression to learn generalized Lotka-Volterra (GLV) model

lambda = [0.01, 0.001]; % penalty parameters for interaction matrix and growth vector

optionsLsqnonlin = optimoptions(...
    @lsqnonlin,...
    'Algorithm', 'levenberg-marquardt',...
    'TolFun', 1e-4,...
    'TolX', 1e-4,...
    'MaxFunEvals', Inf,...
    'Display', 'off'...
    );

%   run linear regression
tic
[optBetaLR, initialGuessDL] = glv_pe_linreg(time, smoothedAbundanceIndex, lowerbound, upperbound, trainSet);
simulatedAbundanceIndexLR   = glv_simulation (time, smoothedAbundanceIndex(1,:), optBetaLR);
sseLR = sum((simulatedAbundanceIndexLR(:) - abundanceIndex(:)).^2);
fprintf('GLV model inferred by linear regression. SSE = %2.2f. Time elapsed = %2.2f s.\n', sseLR, toc);

optBetaLRTable = array2table(optBetaLR, 'VariableNames', [abbrevName, {'Growth'}], 'RowNames', abbrevName);
writetable(optBetaLRTable, './data/glv_coefficients_lr.csv', 'WriteRowNames', true);

%   run latent variable regression
tic;
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
    upperbound,...
    trainSet...
    );
if (exitflag <= 0)
    error('lsqnonlin fails to converge.');
end

%   one last run to obtain optimal GLV parameters
optBetaLVRUnrelaxed = glv_pe_linreg(...
    time,...
    smoothedAbundanceIndex,...
    lowerbound,...
    upperbound,...
    trainSet,...
    'logderiv',...
    reshape(optDL, ns, nt)...
    );
simulatedAbundanceIndexLVRUnrelaxed = glv_simulation (time, smoothedAbundanceIndex(1,:), optBetaLVRUnrelaxed);
sseLVRUnrelaxed = sum((simulatedAbundanceIndexLVRUnrelaxed(:) - abundanceIndex(:)).^2);
fprintf('GLV model inferred by latent variable regression. SSE = %2.2f. Time elapsed = %2.2f s.\n', sseLVRUnrelaxed, toc);

[isStable, largestEig] = check_stability(optBetaLVRUnrelaxed);
if isStable
    fprintf('The unrelaxed GLV model is stable (real part of largest eigenvalue = %2.2f)\n', largestEig);
else
    fprintf('The unrelaxed GLV model is unstable (real part of largest eigenvalue = %2.2f)\n', largestEig);
end

optBetaLVRUnrelaxedTable = array2table(optBetaLVRUnrelaxed, 'VariableNames', [abbrevName, {'Growth'}], 'RowNames', abbrevName);
writetable(optBetaLVRUnrelaxedTable, './data/glv_coefficients_lvr_unrelaxed.csv', 'WriteRowNames', true);

%%  an optional relaxation step to find alternative models with stable coexistence at steady state
%   by resampling GLV parameters in the neighbourhood of the optimal set such that 

nTrials                     = 20;                           % times to resample self-interaction coefficients
resamplesTrials             = zeros(nTrials, ns);           % resampled self-interaction coefficients
sseTrials                   = zeros(nTrials, 1);            % residual as sum of squared difference
relativeAbundanceTrials     = zeros(nTrials, ns);           % relative abundance at steady state
stabilityTrials             = zeros(nTrials, 1);            % stability of equilibrium (unseasible or unstable: false, stable: true)
optBetaTrials               = zeros(nTrials, ns, ns+1);     % optimized GLV parameters

% Levenberg-Marquardt algorithm does not handle constraints
% use trust-region-reflective instead
optionsLsqnonlin = optimoptions(...
    @lsqnonlin,...
    'Algorithm', 'trust-region-reflective',...
    'TolFun', 1e-4,...
    'TolX', 1e-4,...
    'MaxFunEvals', Inf,...
    'MaxIter', Inf,...
    'Display', 'off'...
    );

%   use parallelization
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    distcomp.feature( 'LocalUseMpiexec', false); % seems to reduce time needed to initiate parpool
    parpool('local', 4);
end

poolobj = gcp('nocreate');
if isempty(poolobj)
    poolsize = 0;
else
    poolsize = poolobj.NumWorkers;
end

%   construct a simple parfor wait bar using DataQueue
fprintf('Running parameter relaxation to search for alternative stable models (%d samples, %d workers) ...\n', nTrials, poolsize);
hbar = parfor_progressbar(nTrials, 'Computing ...');  %create the progress bar
tic;
parfor i=1:nTrials
    % resample self-interaction coefficients
    initialBeta = optBetaLVRUnrelaxed;
    resampledDiagonals = zeros(1, ns);
    for j=1:ns
        initialBeta(j,j) = - rand * max(abs(optBetaLVRUnrelaxed(:)));
        resampledDiagonals(j) = initialBeta(j,j);
    end
    resamplesTrials(i,:) = resampledDiagonals;
    
    [optBetaLVRRelaxed_i,~,~,exitflag] = lsqnonlin(...
        @objective_func,       ...
        initialBeta(:),...
        lowerbound,...
        upperbound + 1e-6,...
        optionsLsqnonlin,...
        'beta',...
        lambda,...
        time,...
        smoothedAbundanceIndex,...
        [],...
        [],...
        trainSet...
        );
    if (exitflag <= 0)
        poolobj = gcp('nocreate');
        delete(poolobj);
        error('lsqnonlin fails to converge.');
    end
    
    % optimal GLV parameters
    optBetaLVRRelaxed_i = reshape(optBetaLVRRelaxed_i, ns, ns+1);
    optBetaTrials(i,:,:) = optBetaLVRRelaxed_i;
    
    % residual
    simulatedAbundanceIndexLVRRelaxed_i = glv_simulation (time, smoothedAbundanceIndex(1,:), optBetaLVRRelaxed_i);
    sseTrials(i) = sum((simulatedAbundanceIndexLVRRelaxed_i(:) - abundanceIndex(:)).^2);
    
    % steady state composition
    [stabilityTrials(i), ~, relativeAbundanceTrials(i,:)] = check_stability(optBetaLVRRelaxed_i);
    hbar.iterate(1);   % update progress by one iteration
end
close(hbar);   %close progress bar

% poolobj = gcp('nocreate');
% if ~isempty(poolobj)
%     delete(poolobj);    % close parpool
% end

%   keep only stable models
indexStableSolution         = find(stabilityTrials > 0);
resamplesTrials             = resamplesTrials(indexStableSolution, :);
sseTrials                   = sseTrials(indexStableSolution);
optBetaTrials               = optBetaTrials(indexStableSolution, :, :);
relativeAbundanceTrials     = relativeAbundanceTrials(indexStableSolution, :);
fprintf('Found %d models that predict stable coexistence at steady state.\n', length(indexStableSolution));

%   Get the Opt-fit GLV parameter set with minimal SSE among all stable models
if ~isempty(indexStableSolution)
    [~, indexOptBeta]   = min(sseTrials);
    sseOptRelaxed = sseTrials(indexOptBeta);
    fprintf('The optimal relaxed model obtained. SSE = %2.2f. Time elapsed per sample = %2.2f s.\n', sseOptRelaxed, toc/nTrials);
    
    optBetaOptRelaxed  = squeeze(optBetaTrials(indexOptBeta, :, :));
    optBetaOptRelaxedTable = array2table(optBetaOptRelaxed, 'VariableNames', [abbrevName, {'Growth'}], 'RowNames', abbrevName);
    writetable(optBetaOptRelaxedTable, './data/glv_coefficients_opt_relaxed.csv', 'WriteRowNames', true);
end

%%  figure plot to compare model fitting by different inference algorithms

timeFiner = linspace(time(1), time(end));

%   simulation using GLV inferred from linear regressiion
simulatedAbundanceIndexLR               = glv_simulation (time, smoothedAbundanceIndex(1,:), optBetaLR);
mldLR = fitlm(abundanceIndex(:), simulatedAbundanceIndexLR(:), 'Intercept', false);
simulatedAbundanceIndexLR               = glv_simulation (timeFiner, smoothedAbundanceIndex(1,:), optBetaLR);

%   simulation using GLV inferred from latent variable regressiion\
simulatedAbundanceIndexLVROptRelaxed    = glv_simulation (time, smoothedAbundanceIndex(1,:), optBetaOptRelaxed);
mldLVR = fitlm(abundanceIndex(:), simulatedAbundanceIndexLVROptRelaxed(:), 'Intercept', false);
simulatedAbundanceIndexLVROptRelaxed    = glv_simulation (timeFiner, smoothedAbundanceIndex(1,:), optBetaOptRelaxed);

fprintf('Adjusted R2 of latent variable regression = %2.2f.\n', mldLVR.Rsquared.Adjusted);
fprintf('Adjusted R2 of linear regression = %2.2f.\n', mldLR.Rsquared.Adjusted);

figure();
ylimUB = [3.0, 2.2, 1.6, 1.2, 1.5, 1.0, 1.0, 2.0, 2.0];
for i=1:ns
    subplot(3,3,i);
    hold on;
    plot(time, abundanceIndex(:,i), 'k.', 'MarkerSize',15);
    plot(timeFiner, simulatedAbundanceIndexLVROptRelaxed(:,i),'k-','LineWidth',1);
    plot(timeFiner, simulatedAbundanceIndexLR(:,i),'k--','LineWidth',1);
    box on;
    xlabel('Year');
    ylabel('Abundance index');
    xlim([1992,2016]);
    ylim([0, ylimUB(i)]);
    set(gca,'XTick',[1995,2000,2005,2010,2015]);
    title(char(abbrevName{i}));
    set(gca,'Ticklength',3*get(gca,'Ticklength'));
    legend('Latent variable regression', 'Linear regression');
end
