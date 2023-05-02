# Browser Object Model (BOM)
> Based on the [W3Schools Javascript Tutorial](https://www.w3schools.com/js/default.asp)
> 
This is the model used by the browser to communicate with JavaScript. This is not a standardize model however, almost every browser uses the same properties.

## Window Object
The window object is supported by all browsers and contains all global objects, functions and variables. Global functions are methods of the window object and global variables are properties of the window object.
```javascript
window.document.getElementById(id); // shows that the window contains everything

window.innerHeight // the inner height of the browser window (px)
window.innerWidth  // the inner width of the browswer window (px)
// this is the part of the window that displays the website, not the entire application window

window.open() //opens new window
window.close() // closes current window
window.moveTo() // moves the current window????
window.resizeTo()
```

### Window Screen
```javascript
window.screen // can be written without the window
screen.property

// Screen Properties
screen.width
screen.height
screen.availWidth // avaliable width, the width of screen minus interface width
screen.availHeight // avaliable height, the height of screen minus interface height
screen.colorDepth // number of bits used to display a single colour
screen.pixelDepth // pixel depth of the screen
```

### Window Location
```javascript
window.location.href // returns the href (URL) of the current page
window.location.hostname // returns the domain name of the web host
window.location.pathname // returns the path and filename of the current page
window.location.protocol // returns the web protocol used (http: or https:)
window.location.assign() // loads a new document
```

### Window History
```javascript
history.back() // loads the previous url (browser back button)
history.forward() // loads the next url (browswer forward button)
```

### Popup Boxes
```javascript
window.alert('alert text')
window.confirm('confrimation box')
window.prompt('text', 'default text')
```

### Timing
```javascript
window.setTimeout(function, milliseconds); // function is what occurs at the time out, second arg is the time
window.setInterval(function, milliseconds); // same as setTimeout but runs continously

window.clearTimeout(variable) // this is the variable that the timeout function is set to
window.clearInterval(variable)
```

# Cookies
```javascript
// creating a cookie with an expiry date and path specified (optional)
document.cookie = "username=John Doe; expires=Thu, 18 Dec 2013 12:00:00 UTC; path=/";

// Reading Cookie
let x = doucment.cookie;

// To delete cookie, set expiry parameter to some past date
```