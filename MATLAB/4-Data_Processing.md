# Data Processing in MATLAB

## Preprocessing Data
```matlab
% Method of Processing Data
normalize(x) % normalizes values based on their z-scores
normalize(x, 'center', 'mean') % normalizes values and centres them so the mean is at 0
normalize(x, 'scale', 'first') % normalizes values by the first value of each column in array
normalize(x 'range', [min, max]) % normalizes values into a specified range

% Dealing with Missing Values
isnan(x) % returns logical array where values are NaN
ismissing(x, [var1, var2]) % returns a logical aray where values are NaN or other specified values
mean(x, 'omitnan') % most functions will have a parameter that omits NaN values
standardizeMissing(x, value) % standardizes all the missing value so that the value specified is now equal to NaN
[newarray, missingindex] = rmmissing(A) % removes missing values from array
rmmissing(A, 2) % removes columns with missing values

% You can add "Clean Missing Data" task in the live editor under tasks

% Interpolating Missing Data
[cleanData, missingIndex] = fillmissing(x, 'method') % interpolates missing data using specified method (nearest value, lienear interpolation, etc)
fillmissing = (x, 'linear', 'SamplePoints', [1, 2, 3]) % fillmissing automatically assumes evenly spaced missing values, sample points indicates x values where data is missing/taken
```

## Data Analysis Techniques
```matlab
% Smoothing Data - This can be done with a Task

% Smoothing Data with a Moving Average
smoothdata(y, 'movmean', 3, 'SamplePoints', x) % smooths the data  of y using a moving mean of 3 point intervals using the sample poinnts of x
smoothdata(y, 'method', [b, f]) % smooeths the data for a window that is b moves back from the point and f moves forward from the point

% Linear Correlation - determing how close data correlates linearly

% Plotting the comparitive data points with seprate scaled y-axis can reveal correlation
yyaxis left % sets the active y axis to the left one
yyaxis right % sets the active y axis to the right one (also creates it)

% Correleation Coeffecient
correcoef(v1, v2) % outputs a symmetric matric of coeffecients for different combos of input vectors
correcoef(A, B, 'Rows', 'complete') % ensures that only rows with no missing data are considered

% Polynomial Data Fitting
[c, S, mu] = ployfit(x, y, n) % determines the coeffecients for the best fitting n degree polynomial to the data
% c - polynomial coeffecients, S - used for error estimates in ployval, mu - 2 varible vectors containing mean and std for scaling
polyval(c, x, error, mu) % determintes the outputs of the best fitting polynomial for given x values to calculate
```