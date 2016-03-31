%
% radialpsd - to calculate the radial power spectrum density
%             of the given 2d image
%
% Credits:
% [1] Evan Ruzanski
%     Radially averaged power spectrum of 2D real-valued matrix
%     https://www.mathworks.com/matlabcentral/fileexchange/23636-radially-averaged-power-spectrum-of-2d-real-valued-matrix
%
% Arguments:
%   img  - input 2d image (grayscale)
%   step - radius step between each consecutive two circles
%
% Return:
%   psd     - vector contains the power at each frequency
%   psd_sdd - vector of the corresponding standard deviation
%

function [psd, psd_std] = radialpsd(img, step)
    [N M] = size(img)

    %% Compute power spectrum
    imgf = fftshift(fft2(img))
    % Normalize by image size
    imgfp = (abs(imgf) / (N*M)) .^ 2

    %% Adjust PSD size: padding to make a square matrix
    dimDiff = abs(N-M)
    dimMax  = max(N, M)
    % To make square matrix
    if N > M
        % More rows than columns
        if ~mod(dimDiff, 2)
            % Even difference
            % Pad columns to match dimension
            imgfp = [NaN(N,dimDiff/2) imgfp NaN(N,dimDiff/2)]
        else
            % Odd difference
            imgfp = [NaN(N,floor(dimDiff/2)) imgfp NaN(N,floor(dimDiff/2)+1)]
        end
    elseif N < M
        % More columns than rows
        if ~mod(dimDiff, 2)
            % Even difference
            % Pad rows to match dimensions
            imgfp = [NaN(dimDiff/2,M); imgfp; NaN(dimDiff/2,M)]
        else
            % Pad rows to match dimensions
            imgfp = [NaN(floor(dimDiff/2),M); imgfp; NaN(floor(dimDiff/2)+1,M)]
        end
    end

    % Only consider one half of spectrum (due to symmetry)
    halfDim = floor(dimMax/2) + 1

    %% Compute radially average power spectrum
    % Make Cartesian grid
    [X Y] = meshgrid(-dimMax/2:dimMax/2-1, -dimMax/2:dimMax/2-1)
    % Convert to polar coordinate axes
    [theta rho] = cart2pol(X, Y)
    rho = round(rho)
    i = cell(floor(dimMax/2)+1, 1)
    for r = 0:floor(dimMax/2)
        i{r+1} = find(rho == r)
    end
    % calculate the radial mean power and its standard deviation
    Pf = zeros(2, floor(dimMax/2)+1)
    for r = 0:floor(dimMax/2)
        Pf(1, r+1) = nanmean(imgfp(i{r+1}))
        Pf(2, r+1) = nanstd(imgfp(i{r+1}))
    end

    % adapt to the given step size
    psd = zeros(1, floor(size(Pf, 2) / step))
    psd_std = zeros(size(psd))
    for k = 1:length(psd)
        psd(i) = mean(Pf(1, (k*step-step+1):(k*step)))
        % approximately calculate the merged standard deviation
        psd_std(i) = sqrt(mean(Pf(2, (k*step-step+1):(k*step)) .^ 2))
    end
end

% vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=matlab: %
