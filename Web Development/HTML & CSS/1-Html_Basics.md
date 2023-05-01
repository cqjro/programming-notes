# HTML Syntax Basics
> Based on the [W3Schools HTML Tutorial](https://www.w3schools.com/html/default.asp)

HTML uses tags to identify the structure of the webpage. This is done by creating starting and ending tags which then formats the content present in bewteen them.

```html
<tagname> <!-- Start Tag -->
Heading 1
</tagname> <!-- End Tag -->
```

## URL Basics
> scheme://prefix.domain:port/path/filename


## Basic Document Layout Syntax
```html
<!DOCTYPE html> <!-- defines the file as an HTML5 document -->
<html> <!-- root element of html page, must start and end with it -->
<head> <!-- contains all the meta data of the html page -->
<title> <!-- specifies the title of the webpage (shown in title bar or the page tab)-->
<body> <!-- defines the documents body and containes all the visable elements of the website page -->

<!-- Headngs -->
<h1> <!-- Largest Heading -->
<h2>
<h3>
<h4>
<h5>
<h6> <!-- Smallest Heating -->

<p> <!-- Paragraph -->
<br> <!-- Creates a line break in text -->
<hr> <!-- Creats a line break in text using a horizontal rule -->
<pre> <!-- Displays text as formatted in the html file (includes lines breaks without the use of <br>) -->

<!-- Images -->
<img src="/users/name/documents/image.jpg" alt="Alternative Text - Displays if image is not shown" width="100" height="150">
<img src="img_girl.jpg" alt="Girl in a jacket" style="width: 500px; height: 600px;">
<p><img src="smiley.gif" alt="Smiley face" style="float:right;width:42px;height:42px;">The image will float to the right of the text.</p>
<!-- Float property moves images to the right or left of text-->
<!-- Absolute URL: the exact link from either online or the directory on your computer-->
<!-- Relative URL: the directory relative to the current foloder (begins with forward slash "/")-->

<!-- Links -->
<a href="www.youtube.com"> Link will appear as this text </a>
<!-- can also nest images to hyperlink them -->

<!-- Buttons -->
<button onclick="document.location='default.asp'"> Button Text</button> <!-- Button events are dictated by Javascript code -->

<!-- Bookmarks -->
<h2 id="C4">Chapter 4</h2> <!-- Creates Bookmark -->
<a href="#C4">Jump to Chapter 4</a> <!-- Creates link to jump to bookmark on the same page -->
<a href="html_demo.html#C4">Jump to Chapter 4</a> <!-- Creates link to bookmark on another page -->

```

## Attributes
Attributes can be added to any CSS element in order to provide additional information and are specified in the start tag.
```html
<!-- General Attribute Syntax -->
<p name="value">

<!-- Link Attributes -->
<a href="youtube.com"> <!-- directs the user to the specifed link -->
<a href="mailto:person@email.com">Send email</a>
<!-- Absolute URL: the exact link from either online or the directory on your computer-->
<!-- Relative URL: the directory relative to the current foloder (without the https://www)-->

<a href="youtube" target="_self"> <!-- specifies the location in wihch the link will open>
<!-- _self: Default Opetion that open the link in the same tab that it was accessesed>
<!-- _blank: opens the link in a new window or tab -->
<!-- _parent: opens the link in the parents frame-->
<!-- _top: opens the link in the fully body of the window -->

<!-- Style Attribute -->
<p style="color: blue;"> Paragraph Text </p> <!-- Changes the styling of the element it is placed in-->

<!-- Lang Attribute -->
<html lang="en"> <!-- Specifices the language in which the document is formatted in, in order to aid search engines and browswers -->

<!-- Title -->
<p title="tool tip"> Paragraph Text </p> <!-- Displays a tool tip when the cursor hoevers over the element -->
```

## CSS Styling

### Inline CSS
```html
<tag style="property: value;">
```

### Internal CSS
The syles for specific element types are set for the entrie document when defined in the head 
```html
<!DOCTYPE html>
<html>
<head>
<style>
body {background-color: powderblue;}
h1   {color: blue;}
p    {color: red;}
</style>
</head>
<body>

<h1>This is a heading</h1> 
<p>This is a paragraph.</p>

</body>
</html>
```

### External CSS
A style sheet is defined externally in order to style the website. This is typically used when keeping the same style consistantly for many webpages.
```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="styles.css">
</head>
<body>

<h1>This is a heading</h1>
<p>This is a paragraph.</p>

</body>
</html>
```


### Basic Styling Options
```css
body {
    background-color: red;
}

/* Text Styling */
p {
  color: red;
  font-size: 60px;
  font-size: 130%;
  font-family: courier;
  text-align: center;
  border: 2px solid powderblue;
  padding: 30px; /* defines space between image and border */
  margin: 60px; /* space outside of the border */
}
/* Font Colours can be set with RGB/RGBA, Hex, and HSL/HSLA
Background colours can be specified with RBGA and HSLA, with the A specifying transparency */

/* Link Styles */
<style>
a:link {
  color: green;
  background-color: transparent;
  text-decoration: none;
}

a:visited {
  color: pink;
  background-color: transparent;
  text-decoration: none;
}

a:hover {
  color: red;
  background-color: transparent;
  text-decoration: underline;
}

a:active {
  color: yellow;
  background-color: transparent;
  text-decoration: underline;
}
</style>
```
