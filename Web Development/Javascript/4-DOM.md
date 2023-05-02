# Document Object Model
> Based on the [W3Schools Javascript Tutorial](https://www.w3schools.com/js/default.asp)

This is the hierarchy of elements/objects within an HTML document that is then displayed on a webpage. JavaScript uses this hieracrchy and is able to change the these aspects.

JavaScript can change, add and remove HTML elements/attributes as well as react to events on the page.
JavaScript can additionally change any CSS styling made on the page.

The HTML DOM specicies all elements as objects each with their own properties and methods to access them.

## DOM Methods
```html
<html>
<body>

<p id="demo"></p>

<script>
document.getElementById("demo").innerHTML = "Hello World!"; // replaces the content with id=demo with hello world
</script>

</body>
</html>

getElementById() <!-- is a document method-->
innerHTML <!-- is a property for getting or replacing content of elements-->
```

## Document & Element Methods
### Finding HTML Elements
```javascript
document.getElementsById(id)
document.getElementsByTagName(name) // creates a collection object containing all instances of tag that can be iterated through
document.getElementsByClassName(name)
document.querySelectorAll(tag.class) // finds the elements using the CSS selectors
```
### Finding HTML Objects
```javascript
// These work for a a lot of objects, here are just some examples
document.anchors
document.baseURI
document.body
document.doctype
document.forms
document.head
document.images
document.links
document.scripts
document.title
document.URL
```
### Changing HTML Elements
```javascript
element.innerHTML = "new content"
element.attribute = "new value" 
element.setAttribute(attribute, value) // this is a method the rest are properties 
element.style.property = "new style"
```
### Adding & Deleting Elements
```javascript
document.creatElement(element)
document.removeChild(element)
document.appendChild(element)
document.replaceChild(new_element, old_element)
document.createTextNode("text")
element.remove()
element.removechild()
document.write(text) // write into HTML output
```
### Adding Event Handlers
```javascript
document.getElementsById(id).onclick = function(){code}
```

# HTML Events
HTML events are thigns that happen on in the webpage regarding the html elements which can then trigger JavaScript code to run in response. These events may include, user-input or browser actions (Page finished loading).

## Basic Syntax
```html
<element event="*Insert JavaScript Code*">
<button onclick="document.getElementById('demo').innerHTML = Date()">The time is?</button>
```

## Common HTML Events
```html
onchange <!-- HTML Element has been changed-->
onclick 
onmousedown
onmouseup
onmouseover
onmouseout
onkeydown
onload
onunload
```
## Event Listeners/Handlers
Event capturing works by considering the outer most elements event first then works its way inwards, where was event bubbling starts from the inner most elements events and bubbles outwards.
```javascript
element.addEventListener(event, function, useCapture) //useCapture is a boolean for using event bubbling or event capturing, this is to control order of event handling
// The addEventListener() method adds an event listener without overwriting ones that already exist

window.addEventListener("resize", function(){           // you can add event listeners to the window, document, xmlHTTPRequest or other event supporting objects
  document.getElementById("demo").innerHTML = sometext;
});

element.removeEventListener(event, function) // safely removes existing event listeners
```

# Navigation
HTML DOM allows for navigation using the nodal relationship of the hierachy. The document itself is the starting **document node** with every HTML element being subsequent element nodes. The text within the the element notes are denoted as **text nodes** and every attribute is an **attribute node**. Comments are also included and are denoted as **comment nodes**.
```javascript
// Navigation Properties
.parentNode
.childNodes[nodenumber]
.firstChild
.lastChild
.nextSibling
.previousSibling
.nodeValue// returns node value, returns null for element nodes
.innerHTML // for text nodes, same as accessing the node value of the first child

.nodeName // returns the name of the current node (same as tag name)
.nodeType // returns the type of node

document.body // access to the entire body 
document.documentElement // access to the full document
```
## Creating Nodes
```javascript
const paragraph = document.createElement("p")
const node = document.createTextNode("New Text")
paragraph.appendChild(paragraph)
```