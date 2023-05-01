# MATLAB Basics

## Variables, Arrays & Vectors
```matlab
% Variables - (starts with a letter, contains only letters, numbers and underscores)
m = 3*5; % semicolons stop the result from displaying but will be visable in the variable workspace

% Arrays
A = [3 5; 6 7] % rows are seperated by spaces
>>> 3  5
    6  7

% Regularly Spaced Vectors Method 1: Colon Operator
y = 1:210 % creates a vector from 1-10 with a step of 2 (default step of 1 if not specified)
>>> 1 3 5 7 9

% Regularly Spaced Vectors Method 2: Linspace Function
y = linspace(first, last, # of elements) % returns a vector with the number of elements specified within the given range (infered step)
y = linspace(1, 10, 5)
>>> 1.0000 3.2500 5.5000 7.7500 10.0000 

% Rearranging Matricies
reshape(A, n, m) % reshapes the matrix A into an nxm matrix (total numbers of elements must be the same)
reshape(A, n, []) % reshape automatically fills in the missing dimension when [] is input
A(:) % reshapes any dimission 
```

## Files & Command Prompts
```matlab
% Command Prompt & Workspace
clear % clears workspace of variables 
clc % clears command space
format long % formats the results to have longer decimal values
format short % formats decimals to have shorter decimal values (4 decimal places I believe)

% Files & Data
save filename.mat % save all variables (in the command promt) to a file of the name specified
load filename.mat  % loads the data from a mat file
load filename.mat m % loads only the variable k from the file

% Table Maniplulation of Data
load datafile % loads csv, mat or other file
clf
csvread('file.csv')
table = readtable('file.csv')
values % displays data in the flie as a table
d = vales.Density % saves all the values in the Density column of the values table into a variable
elements.new = [] % adds new row to the table with name new (must be n-length column vector same size as table)
table(A, B) = converts arrays into table data types
array2table(A) = converts an array to a table

% Sorting Data
values = sortrows(values, 'new') % sorts rows from smallest to largest in the 'new' column
values = sortrows(values, 'new', "descend") % sorts rows from largest to smallest in the 'new' column
% Note: if you manipulate the data in the table manually, there should be a prompt to update the code
```

## General Functions
```matlab
% ** specific values for mutiple output functions can be ignored by using ~
	% Example:
		[~, ind] = max(A) % only applies the index value to ind variable

% Basic Usefult Fucntions
rand(n, m) % creates an nxm matrix random numbers
randn(n, m) % creates an nxm maxtric of normally distrubuted random numbers
randi(n, m) % creates an nxm matrix of uniformly distrubuted random numbers

zeros(n, m) % creates an nxm matrix of zeros
ones(n, m) % creates an nxm matrix of ones
size(A) % returns the size of an object A, (returns vector of dimensions for arrays)
	-> [#row, #col] = size(A)
max(A) % returns the maximum value in the array
	-> [Max, index] = max(A)
round(A) % round the elements of A
isnan(A) % returns boolean matrix for values that are NaN
nnz() % number non-zero returns the number of non-zero values
eye(n) % creates an nxn identity matrix
diag([1, 2, 3]) % creates a diagnoal matrix with a given vector input
datetime(year, month, day, hour, minute, seconds) % creates a datetime data type
days(time) % converts a time value into a number of days
categorical(stringArray, [1, 2, 3], levels, 'Ordinal', true) % converts a string array into a categroical one
% levels define threshhold for values to be considered part of a category, ordinal specifiese there is a order (small, med, large))
categories(catArray) % shows all the UNIQUE categories in the catArray
summary(catArray) % shows the number of each category in the array
mergecats(catArray, [cat1, cat2], new_cat) % merges categories in the cat array

% Note: Functions like std, min, max, mean will perform the operation for each column unless specified otherwise
```

## Matrix Operations
```matlab
% Operations
x = A[1 2]
y = x + 2 % adds to to every value in the array
y = x*2 % multiplies every value in the array by 2

% Array Operations
A = [1 2]
B = [2 3]

A + B % adds the elements of the vectors element wise
A.*B % performs element wise multiplaction of arrays (for n_A = n_B or m_A = m_B or both)
A*B % performes matrix multicplation for n_A = m_B or vise versa

% Matrix Contatination
[A; B] % combines the two matrixes vertically
[A B] % combines the matrixes horizontally

% Matrix Functions
rref(A) % performs gaussian elimation on the given matrix
linsolve(A, B) % solves the system Ax = B
x = A\B % solves the system Ax = B
det(A) % returns the determinant of a give matrix
x = x' % transposes the vector/matrix
inv(A) % returns the inverest of a matrix
rank(A) % returns the rank of a matrix (maximum amount of linearly independent columns)
```

## Indexing & Boolean Operators
```matlab
% Indexing Matrcies
A = [1 2 3; 4 5 6]

A(row, col) % returns the value of the specified index in the matrix
A(row, :) % returns the entire row of the matrix
A(:, col) % returns the entire column of a matrix
A(1:4, col) % returns the values in the column from rows 1 to 4
A(end, col) % returns the last value in the given column
A(end-1, col) % returns the second last element of the given column
A([1 3 6]) % returns the 1st, 3rd and 6th value of the matrix A

% Boolean Operators
& % and
~ % not
| % or

% Logical Indexing
A > 4 % returns a nxm matrix of 0s and 1s indicationing true or false values of the logic
B = A(A > 4) % creates a matrix B that contains all the elements in a that are greater than 4
D = C(A > 4) % returns a matrix of values corresponding to the indexes of true values of A > 4
A(A > 4) = 0 % reassigns the values where A > 4 to 0
```

## Conditional Statements & Loops
```matlab
% if Statement Syntax
if condition == True
	code
elseif condition2 == True
	code2
else
	code3
end

% For Loop Syntax
for x = 1:5
	code
end
```