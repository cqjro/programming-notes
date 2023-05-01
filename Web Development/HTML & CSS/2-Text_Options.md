# Text Formatting Options
> Based on the [W3Schools HTML Tutorial](https://www.w3schools.com/html/default.asp)

## Basic Text Options
```html
<b> <!-- Bolds Text -->
<strong> <!-- Importnat Text -->
<i> <!-- Italic Text -->
<em> <!-- Emphasized Text -->
<mark> <!-- Marked Text (Highlighted)-->
<small> <!-- Small Text -->
<del> <!-- Deleted Text (Striked Through) -->
<ins> <!-- Interted Text (Underlined)-->
<sub> <!-- Subscript -->
<sup> <!-- Superscript -->
<bdo> <!-- Bi-Directional Overide, causes text to be written from right to left (mirrors text)>
```

## Quotation Elements
Quation elements are used for sections of the website that are taken from another source and need to be referenced.
```html
<blockquote cite="websitesource.com"> <!-- indents a block of text that is quoted from anothe source -->
<q> <!-- Short inline text quotes -->
<abbr title="As Soon As Possible"> <!-- creates better accesiblity and helps search engines-->
<address> <!-- defines the contact information of the owners of the website -->
<cite> <!-- defines the title of something being cite, rendered in italics usually>
```

## Table Formatting
```html
<table> <!-- Defines the table -->
  <caption> Table Caption/Title </caption>
  <tr> <!-- Defines the table row -->
    <th> Country </th> <!-- Defined the table heading -->
    <th> GDP </th>
    <th> Population </th>
  </tr>
  <tr>
    <td> Canada </td> <!-- Defines Table Data-->
    <td> $2 </td>
    <td> 39.5 Million </td>
  </tr>
</table>

<th colspan="2"> <!-- Lets the cell span the specified number of columns -->
<th rowspan="2"> <!-- Lets the cell span the specified number of columns -->
<colgroup>
  <col span="2" style="style"> <!-- groups together columns for specific styling -->
</colgroup>
```
### Table Styling
```css
/* Table Styling */
table, th, td {
  border: 1px solid black; /* Sets border for all cells in the table */
  border-collapse: collapse; /* Prevents the formation of a doubel border */
  border-radius: 10px; /* Changes cells to be round */
  border-style: dotted; /* Changes line style of border */
  border-color: #96D4D4;
  background-color: #96D4D4;
  width: 100%; /* fits page size, works for cells as well */
  height: 400px; /* changes height of cells */
}
```

## Lists
```html

<!-- Unordered List -->
<ul>
  <li> Item 1 </li>
  <li> Item 2 </li>
    <ul> 
      <li> Item 1a </li>
    </ul>
  <li> Item 3 </li>
</ul>

<!-- Ordered List -->
<ol type = "A"  start = "D"> <!-- Type attribute changes list from numbers to letters and roman numerals, start changes starting point -->
  <li> Item 1 </li>
  <li> Item 2 </li>
  <li> Item 3 </li>
</ol>

<!-- Descriptive List -->
<dl>
  <dt> Coffee </dt> <!-- Description Title-->
  <dd>- black hot drink</dd> <!-- Descrition Description -->
  <dt> Milk </dt>
  <dd> - white cold drink </dd>
</dl>
```
### List Styling Options
```css
ul {
  list-style-type: disc; /* changes the list bullet type */
}

li {
  float: left; /* List Elemenets are commonly floated left to create naivation menu bars */
}
```

## Character Entities
HTML reserved characters that cannot be regularly displayed in text. As a result, they are represented using entities.
```html
<	less than	                          &lt;	  &#60;	
>	greater than	                      &gt;	  &#62;	
&	ampersand	                          &amp;	  &#38;	
"	double quotation mark	              &quot;	&#34;	
'	single quotation mark (apostrophe)	&apos;	&#39;	
¢	cent	                              &cent;	&#162;	
£	pound	                              &pound;	&#163;	
¥	yen	                                &yen;   &#165;	
€	euro	                              &euro;	&#8364;	
©	copyright	                          &copy;	&#169;	
®	registered trademark	              &reg;   &#174;

 ̀	a	a&#768;	à	
 ́	a	a&#769;	á	
̂	a	a&#770;	â	
 ̃	a	a&#771;	ã	
 ̀	O	O&#768;	Ò	
 ́	O	O&#769;	Ó	
̂	O	O&#770;	Ô	
 ̃	O	O&#771;	Õ

```


