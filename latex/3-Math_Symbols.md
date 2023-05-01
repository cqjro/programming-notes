# Latex Math Symbols
> Note: All images are from The Math Respository's Latex tutorial compilation video.
> [Click Here](https://youtu.be/lRD_7PSiRAc) to watch the video.

# Math Setup
```latex
% Types of Math Modes

% Display Style - Standalone equations from text
\[f(x) = (x+2)^2 -9\] % \[ \] is used to displayed single line equations

\begin{align*} % aligns the equations at the &
f(x) & = x^2
f'(x) & = 2x
\end{align*}
% Note: using align without the * will result in the equations being automatically numbered


% Inline/Text Style - Math as a part of text paragraph (usually for variables)
\( f(x) = x^2 \) % for inline math (also can use $ x^2 $)

\[ \textstyle <<< \] % changes the displaystyle into inline math (works the other way)
```

## Basic Mathematical Symbols
```latex
% Basic Mathimatical Notation

5 + 3
5 - 3
5 \cdot 3 % dot product
5 \times 3 % cross product
27 \div 9 % division sign (kinda shit for displaying math)
\sqrt[n]{} % radical with optional argument for the nth root
\pm \mp % plus and minus symbol
\geq \leq % greater than or equal to / less than or equal to
\neq % not equal to
\approx
\equiv % three bar equals sign

\frac{numerator}{denominator}
\dfrac % display style fraction
\tfrac % inline style fraction

a^{23} % super script
a_{bc} % subscript

\left \right % latex automatically chooses appropriate size brackets for equation
% there are ways of getting specific bracket sizes but like why would you

\text{text} % inserts text into the math

\delta % lowercase greek letter
\Delta % uppercase greek letter

AMS symbol package -> \mathbb{R} % double lined letters (ex. real numbers)
					  \mathfrak{p} % the goofy ah new york times ass letters (fraktor)
```

## Advanced Mathemetical Symbols
```latex
% Overset & Underset
\overset{?}{=} % places to things above or below eachother
\underset{=}{?}
% Note both of these can be used at the same time


% Trig functions
\sin(x)
\cos(x)
\tan(x)

% Other functions
\ker{x}

% Custom functions
\operatorname{function}(x)

% Limits
\lim_{x \to \infinity}

% Sigma & Pi notation
\sum_{n=1}^{\infinity} \frac{1}{n}
\prod_{n=1}^{\infinity} \frac{1}{n}

% Integrals
\int % single integral
\iint % doulbe integral
\iiint % triple integral
\oint % integrals across an entire surface

\int_0^\ininitfy % integral from 0 to inifinity

\iiint f(x,y,x) \, dx \, dy \. dz % creates spaces in the information so the information isnt akward

F(x) \bigg\vert_a^b % creates the bar from evaulated integrals


% Partical derivatives
\frac{\partial f}{\partial x} % creates the partial deritavtive symbol
\nabla 

% Vectors
\vec{x} % built in vector notatiojn
\vv{x} % esvect package which stretches the arrow accros the entire variable


% Special variables
\dot{x} % creates the dot (per second) variables 
\hat{x} % creates the hat (per mass/mols) variables

% Set Theory
\{ x \in \mathbb{R} \mid x > 0 \}  % \in is epsilon and \mind is the "such that"

\subset \subseteq \supset \supseteq
\cup \cap % union between sets
\emptyset \varnothing

% Logic Symbols
\land \lor \lnot % logical operators
\implies \impliedby \iif % implications
\forall \exists % quantifiers
\therefore
```

## Tables & Arrays
```latex
% Tables - Arrays of Strings (what you would expect)
\begin{table} % tables in academic writting like table 1.1
\caption{table title} 
\begin{tabular}{|l|c||r} % the number of letters = number of columens, | = bars that seperate columns
% c = centre, l = left justified, r = right justified
\hline % creates a horizontal line for the next row
text & text & text \\ % new columns are indicated by &, new rows are indicated by \\
\end{tabular}
\end{table}

\usepacke{hhline} -> \hhline{|-|=|} % hhline package lets you be more specific with the table lines
\multicolumn{num of col}{alignment of new}{contents}
\multirow[valign]{num of rows}{width}{contents} % requires multirow package

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Array - Matricies
% Note: Must be called in math mode
% Same table commands apply

% normal
\left[ \begin{array}{ccc}
a & b & c \\
d & e & f
\end{array} \right[ %closes the matrix in square brackets

%AMS Math Package
\begin{bmatix} % use pmathix for circle brackets, no need to specifiy columns
a & b & c
\end{bmatix}
```

<center>
<img src=/images/pmatrix.png width=700>
</center>
