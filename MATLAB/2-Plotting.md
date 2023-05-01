# MATLAB Plotting

## 2D Plotting Basics
```matlab
% Simple Data Plots
v1 = [1; 2; 3; 4; 5; 6; 7;]
plot(x, y) % plots the data of an x array and y array
plot(v1) % plots the data of the vector v1 as the y-axis and uses 1 -> n has the x-axis
scatter(x, y) % plots points without lints filled in


% Line Specifications
plot(x, y, "r--o") % plots the graph with a red dashed line and circle point markers

% Line Styles
- % solid line
-- % dashed line
: % dotted line
-. % dashd-dotted line

% Common Markers
o % cricle
+ % plus sign
* % aesterix
. % point
x % cross
_ % horizontal line
| % veritcal line

% Colour Short names
r % red
g % green
b % blue
c % cyan
m % magenta
y % yellow
k % black
w % white

% Plotting multiple lines to the same graph
hold on % plots after this line will plot to the most recent plot created
hold off % plots after this point will create a new plot

% Additional Arguments
plot(v1, 'LineWidth', 3) % sets the plot line width to 3, additional arguments can be changed with subsequent values afterwards

% Additional Functions
title('title') % adds the title to an existing plot
ylabel('label') % adds y-axis label to existing plot
xlabel('label') % adds x-axis label to existing plot
legend('line1', 'line2') % adds legend with labels coressponding to the order data was added to the plot
axis % no inputs returns the current x and y limits of the current plot
axis tight % changes the limits to match the largest and smallest values of data provided
ylim([lower upper]) % changes the y-limits
xlim([lower upper]) % changes the x-limits
polyfit(x, y, n) % attempts to creata a trendline to the data based on an n degree polynomial
```

## Types of 2D Plots
```matlab
plot(x, y) % regular plots
scatter(x, y) % point plots
histogram(x, y) % bar graphs
plotmatrix(A) % plots all of the columns of a matrix against eachother
```

## 3D Plots
```matlab
% 3D Plots
surf(A) % represents an nxm matrix by using the stored values as the height and the n & m as the x and y values
mesh(A) % same thing as surface plot but plots a mech of lines with unfilled surfaces
contour(A) % visualizes the 3D plot as a 2D one (level curve type beat but not really)

% Functions
yticks(1:12) % sets where ticks are places on axis
yticklabels(M) % sets labels to months of the year
ytickangle(30) % changes the angle that ticks are places around the axis
```