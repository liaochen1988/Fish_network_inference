%   This function smooths time series data using empirical mode decomposition
%   Last modified by Chen Liao on May 17, 2019

function smoothedAbundance = smoothing(time, abundance, train)

if (size(abundance) ~= size(train))
    error('dimension of y and train are not equal.');
end
[nt, ns] = size(abundance);

%   interpolation of missing data (data classified to test set) along time series
yInterp = abundance;
for i=1:ns
    yInterp(:,i) = pchip(time(train(:,i)) , abundance(train(:,i) , i), time); % spline interpolation
    
    %   correct for negative values
    if (sum(yInterp(:,i) <= 0) > 0.5)
        
        indexFirstPositiveValue = find(yInterp(:,i) > 0, 1); % find the index of the first value that is above zero
        
        %   for elements from indexFirstPositiveValue + 1 to end
        %   replace any value below zero with its nearest positive neighbour from the left side
        for j=indexFirstPositiveValue+1:nt
            if (yInterp(j,i) <= 0)
                yInterp(j,i) = yInterp(j-1,i);
            end
        end
        
        %   for elements from 1 to indexFirstPositiveValue - 1
        %   replace any value below zero with its nearest positive neighbour from the right side
        for j=indexFirstPositiveValue-1:-1:1
            if (yInterp(j,i) <= 0)
                yInterp(j,i) = yInterp(j+1,i);
            end
        end
    end
end

%   empirical Mode Decomposition
smoothedAbundance = yInterp;
for i=1:ns
    IMF     = emd(yInterp(:,i))'; % intrinsic mode function
    Trend   = yInterp(:,i);
    
    %   keep trends with Hurst exponent > 0.5
    for j=1:size(IMF,2)
        hurst_expo = estimate_hurst_exponent(IMF(:,j)');
        if(hurst_expo < 0.5)
            Trend=Trend - IMF(:,j);
        end
    end
    smoothedAbundance(:,i) = Trend;
    
    %   replace values smaller than zero with original data
    for j=1:nt
        if (smoothedAbundance(j,i) <= 0)
            smoothedAbundance(j,i) = yInterp(j,i);
        end
    end
end

end

