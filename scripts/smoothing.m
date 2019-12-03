%   This function smooths time series data using empirical mode decomposition
%   Last modified by Chen Liao on Dec 3, 2019

function smoothedAbundance = smoothing(time, abundance)

[nt, ns] = size(abundance);
assert(nt == length(time));

%   Empirical Mode Decomposition
smoothedAbundance = abundance;
for i=1:ns
    IMF     = emd(abundance(:,i))'; % intrinsic mode function (IMF)
    Trend   = abundance(:,i);
    
    %   keep IMFs with Hurst exponent >= 0.5
    for j=1:size(IMF,2)
        hurst_expo = estimate_hurst_exponent(IMF(:,j)');
        if(hurst_expo < 0.5)
            Trend = Trend - IMF(:,j);
        end
    end
    smoothedAbundance(:,i) = Trend;
    
    %   replace values smaller than zero with original data
    for j=1:nt
        if (smoothedAbundance(j,i) <= 0)
            smoothedAbundance(j,i) = abundance(j,i);
        end
    end
end

end
