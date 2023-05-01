# Block & Inline Elements
> Based on the [W3Schools HTML Tutorial](https://www.w3schools.com/html/default.asp)

## Block-Level Elements
Block level elements are elements that:
1. Start on a new line
2. The browser automatically adds a margin before and after the element
3. Takes up the full width of the space provided

### Common Block-level Elements:
```html
<p> <!-- Most Common -->
<div> <!-- Most Common -->
<address>
<article>
<aside>
<blockquote>
<canvas>
<dd>
<dl>
<dt>
<fieldset>
<figcaption>
<figure>
<footer>
<form>
<h1>-<h6>
<header>
<hr>
<li>
<main>
<nav>
<noscript>
<ol>
<pre>
<section>
<table>
<tfoot>
<ul>
<video>
```

## The ```<div>``` Element
The ```<div>``` element is used as a container to define the different sections of the webpage. It does this through styling the size, colour, and padding.
Additionally, these are typically styled using class defintions and bookmarked as these are major sections of the webpage.

## Inline Elements
These are elements that do not start on a new line and only take of as much horizontal space as necessary.
### Common Inline Elements
```html
<span> <!-- Most Common -->
<a>
<abbr>
<acronym>
<b>
<bdo>
<big>
<br>
<button>
<cite>
<code>
<dfn>
<em>
<i>
<img>
<input>
<kbd>
<label>
<map>
<object>
<output>
<q>
<samp>
<script>
<select>
<small>
<strong>
<sub>
<sup>
<textarea>
<time>
<tt>
<var>
```
## The ```<span>``` Element
This is used as an inline container for text to apply specific stylings or bookmark important points in the text.

# Classes in HTML
Classes are used to specifiy the class of **ANY** element in HTML. These classes can then have associated stylings or can be manipulated using Javascript.

## Class Syntax
```html
<head>
    <style>
    .class1{
        background-colour: white;
    }
    </style>
</head>

<h2 class="class1 class 2 class3">Text</h2>
```

## Class Examples
```html
<!--Class Examples  -->

<!-- Example #1  -->
<!DOCTYPE html>
<html>
<head>
<style>
.city {
  background-color: tomato;
  color: white;
  border: 2px solid black;
  margin: 20px;
  padding: 20px;
}
</style>
</head>
<body>

<div class="city">
  <h2>London</h2>
  <p>London is the capital of England.</p>
</div>

<div class="city">
  <h2>Paris</h2>
  <p>Paris is the capital of France.</p>
</div>

<div class="city">
  <h2>Tokyo</h2>
  <p>Tokyo is the capital of Japan.</p>
</div>

</body>
</html>

<!-- Example #2-->
<!DOCTYPE html>
<html>
<head>
<style>
.note {
  font-size: 120%;
  color: red;
}
</style>
</head>
<body>

<h1>My <span class="note">Important</span> Heading</h1>
<p>This is some <span class="note">important</span> text.</p>

</body>
</html>
```

# Bookmarks/IDs
IDs are similar to classes, as they define the specific styling of elements, however they are unique and can only refer to one element instead of any element.
```html
<!DOCTYPE html>
<html>
<head>
<style>
#myHeader {
  background-color: lightblue;
  color: black;
  padding: 40px;
  text-align: center;
}
</style>
</head>
<body>

<h1 id="myHeader">My Header</h1>

</body>
```