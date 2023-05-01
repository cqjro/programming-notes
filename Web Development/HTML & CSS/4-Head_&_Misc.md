# HTML Head Options
> Based on the [W3Schools HTML Tutorial](https://www.w3schools.com/html/default.asp)

```html
<head>
<!--Page Title-->
<title> Home Page </title>

<!-- Favicon -->
<link rel="icon" type="image/x-icon" href="/images/favicon.ico">

<!-- External CSS Styling -->
<link rel="stylesheet" href="styles.css">

<!-- Internal CSS Styling -->
<style>
body {background-color: powderblue;}
h1   {color: blue;}
p    {color: red;}
</style>

<!-- Meta Data --> 
<!-- Meta Data is not displayed on the website but is used by browswers and search engines for how content should be displayed or found-->
<meta charset="UTF-8">
<meta name="keywords" content="HTML, CSS, JavaScript"> <!-- For search engines -->
<meta name="description" content="Website does stuff">
<meta name="author" content="John Doe">
<meta http-equiv="refresh" content="30"> <!-- Refreshes content every 30 seconds -->
<meta name="viewport" content="width=device-width, initial-scale=1.0"> <!-- Sets the viewpoint based on the device used-->

<!-- Base URL -->
<base href="https://www.w3schools.com/" target="_blank">
<!-- Sets he base URL for all URLs on the page--->

</head>
```

# Miscellaneous Stuff

### Iframe
This allows you to display a seperate webpage inline with the current one.
```html
<!-- Syntax -->
<iframe src="url" title="description"> </iframe>

<!--Iframe as Link Target -->
<iframe src="demo_iframe.htm" name="iframe_a" title="Iframe Example"></iframe>
<p> <a href="https://www.w3schools.com" target="iframe_a">W3Schools.com</a> </p>
```

### Javascript
This allows you to run scripts written in Javascript either inline with the document or from a linked file
```html
<script>
document.getElementById("demo").innerHTML = "Hello JavaScript!";
</script>
<noscript>Sorry, your browser does not support JavaScript!</noscript>
```
The ```<noscript>``` tag defines the content that will be displayed if scripts are disabled
