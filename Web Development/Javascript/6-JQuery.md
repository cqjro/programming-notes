# JQuery
> Based on the [W3Schools Javascript Tutorial](https://www.w3schools.com/js/default.asp)

JQuery is JavaScript library used to handle browser incompatibilites and helps simplfiy the HTML DOM.

## Finding Elements
```javascript
myElement = $("#id01") //JQ
myElement = document.getElementById("id01"); // JS

myElements = $("p"); // JQ
myElements = document.getElementsByTagName("p"); // JS

myElements = $(".intro"); //JQ
myElements = document.getElementsByClassName("intro"); //JS

myElements = $("p.intro"); //JQ
myElements = document.querySelectorAll("p.intro") //JS
```

## HTML Elements
### Text Content
```javascript
myElement.text("Hello Sweden!"); // JQ
myElement.textContent = "Hello Sweden!"; //JS

myText = $("#02").text(); // JQ
myText = document.getElementById("02").textContent; // JS
```
### HTML Content
```javascript
myElement.html("<p>Hello World</p>"); // JQ
myElement.innerHTML = "<p>Hello World</p>"; // JS

content = myElement.html(); // JQ
content = myElement.innerHTML; // JS
```

## CSS Styles
```javascript
myElement.hide(); // JQ
myElement.style.display = "none"; // JS

myElement.show(); // JQ
myElement.style.display = ""; // JS

$("#demo").css("font-size","35px"); // JQ
document.getElementById("demo").style.fontSize = "35px"; // JS
```

## HTML DOM
```javascript
$("#id02").remove(); // JQ
document.getElementById("id02").remove(); // JS

myParent = $("#02").parent().prop("nodeName"); // JQ
myParent = document.getElementById("02").parentNode.nodeName; // JS
```