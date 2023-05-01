# Text & Environments
> Note: All images are from The Math Respository's Latex tutorial compilation video.
> [Click Here](https://youtu.be/lRD_7PSiRAc) to watch the video.

## Font & Text Editing
```latex
% How to use Font Size Commands -> extsize, anyfontsize for more sizes
Normal size text \LARGE LARGE TEXT % everything after text will be large
Normal size test {\LARGE LARGE} test % everything in the brackets will be large


% Font Styles - Work as functions for specific text
\textbf{Bold text}
\textit{Italic text}
\underline{Underlined text} % does not work across multiple lines very well

ulem package -> \uline{text}, \uuline{doulbe line}, \uwavy{wavey underline}

\emph{text \emph{text} text} % automatically emphasizes text based on context (dont really get it)


% Font Family Switching
\textrm{Roman text}  % default font
\textsf{Sans serif text}
\texttt{typewriter text} % for monospaced needs


% Justifiying text - word spacing is not stretched out
\noindent % new parapgraphs are automatically indented in latex, this stops it
\begin{center} text \end{center} % centres text
\begin{flushleft} text \end{flushleft} % left justifies text
\begin{flushRight} text \end{flushLeft}
% Note: use \\ to add lines withing these justifications 
% \\[spacing] will justify the spacing Ex. \\[\baselineskip]


% Controlling Spacing
the real number~$x$ % this ensures that x is linked to the text so it stays on the same line
\hspace{length} % creates a horizontal space of a given length (1cm)
\underline{\hspace{6cm}} % creates a 6cm line
\vspace{length}  % creates a vertical space of indicated length
\phantom{text} % creates invisable text to help with alignment

% Other formatting
\fbox{} % creates a box around text
\boxed{} % creates a box around objects in mathmode
\noindent\hrulefill % fills the rest of the line with a solid line
\rule{width}{height} % creates a line with specified dimensions
```

## Font Size Changing
<center>
<img src=/images/fontsize.png width=700>
</center>

## Images
```latex
\usepackage{graphics}
\includegraphics[width=6cm]{image.jpg} % works for jpg, png, eps, pdf
```
## Itemized & Numbered Lists
<center>
<img src=/images/lists.png width=700>
<img src=/images/custombullets.png width=700>
<img src=/images/enumitem.png width=700>
</center>

## Theorem Environments
<center>
<img src=/images/asmath.png width=700>
<img src=/images/pythagorus.png width=700>
<img src=/images/theoremstyles.png width=700>
</center>

> Note: You can create your own theorem styles

## Proof Environments
<center>
<img src=/images/proofs.png width=700>
</center>

## Custom Environments
<center>
<img src=/images/customenvironments.png width=700>
</center>

## Custom Commands
```latex
\newcommand{\new}{action} % must be placed in the preamble of the document

% Example
\newcommand{\RR}{\mathbb{R}} % shortcut for the real numbers

\newcommand{\myoperator}{\operatorname{myoperator}
\DeclareMathOperator{\myoperator}{my operator}

\DeclareMathOperator{\Fun}{Fun} % creates a fun function
\DeclareMathOperator*{\Fun}{Fun} % creates a fun function where superscrupts and subscripts go over and under the function
```

<center>
<img src=/images/functions.png width=700>
</center>
