# Basics of Javascript
> Based on the [W3Schools Javascript Tutorial](https://www.w3schools.com/js/default.asp)

Javacscipt code is run either in the script tag of an HTML document or called from in a seperate file, where scripts can eiher be placed in the head or body.
```html
<script>
    document.getElementById("demo").innerHTML = "My First JavaScript";
</script>

<script src="test.jg"> </script>
```

## Function Definition
Functions can either be called by the script itself, or when an event occurs, such as when a button is pushed.
```html
<script>
    function exampleFunction(param1, param2) {
        document.getElementId("test").style.display = "none";
    }
</script> 

<button type="button" onclick="exampleFunction()"> Make stuff disappear </button>
```

## Displaying Javascript Output
### Writing to an Element: ```innerHTML```
```html
<script>
    // This changes the content of the element with the ID "demo" to "My First JavaScript" 
    // by directly writing into the element
    document.getElementById("demo").innerHTML = "My First JavaScript";
</script>
```
### Wrinting to HTML Output: ```document.write()```
```html
<script>
    // Writes test to the html page when placed in the body
    // Apparently using this after the HTML page has lodeded will delete all existing HTML?
    // This is only used for testing
    document.write("test");
</script>
```

### Window Alerts: ```window.alert()```
```html
<script>
    window.alert("WARNING THIS IS A VIRUS")
</script>
```

### Output to Browser Console: ```console.log()```
```html
<script>
    console.log("Hello World")
</script>
```

## Defining Variables
```javascript
// Method #1
// These variables are block scoped
// The variables can be updated but not redeclared
let x; // declares varibale
x = 2; // assigns value to variable

// Method #2
// These variables are block scoped
// The variables cannot be updated or re-declared
const x;
x = 2;

// Method #3
// These variables are globlly or functionally scoped
// These varibales can be updated and redeclared with its scope
var x = 2;
```

## Operators
```javascript

// Arithmatic Operators
+ - * / ** %

++ // increment
-- // decrement

+= -= *= /= etc

// Comparison Operators
== // equal to
=== // equal to and equal type???
!= // not equal
!== // not equal value or type
> // greater than
>= // greater than or equal to
< // less than
<= // less than or equal to
? // ternary operator

// Logical Operators
&& // and
|| // or
! // not

// Type Operators
typeof // type of variable
instanceof // returns true if object is an instance of an object type?
```

## String Methods
```javascript
text.length // gets the length of the string
text.slice(1, 5) // returns part of the string from the start to end position (end not included)
                // starting index at 0
text.substr(1, 7) // returns part of the string but the argumnets are the start position and the length
text.replace("Replace Me", "New Text")

// I am too lazy you'll end up searching these up anyways

// Template Literals
let test = ` The values of x is ${x}`;
let html = `<h2>${header}</h2><ul>`;
```

## Date Objects
Date objects are static and do not change like a ticking clock. They are set based on the moment they are defined.
```javascript
const d = new Date() // Current Date
const d = new Date(2023-02-02) // Creating a specific date
```

## Javascript Math Object
The Math object is a static object that does not require a constructor to use all methods avaliable
```javascript
// Constants
Math.E        // returns Euler's number
Math.PI       // returns PI
Math.SQRT2    // returns the square root of 2
Math.SQRT1_2  // returns the square root of 1/2
Math.LN2      // returns the natural logarithm of 2
Math.LN10     // returns the natural logarithm of 10
Math.LOG2E    // returns base 2 logarithm of E
Math.LOG10E   // returns base 10 logarithm of E

// Methods

Math.round(x)	// Returns x rounded to its nearest integer
Math.ceil(x)	// Returns x rounded up to its nearest integer
Math.floor(x)	// Returns x rounded down to its nearest integer
Math.trunc(x)	// Returns the integer part of x 
Math.pow(x, y)  // Power
Math.sqrt()
Math.abs()
Math.sin()
Math.cos()
Math.min()
Math.max()
Math.random()
Math.log()
```

## Arrays & Objects
```javascript
// Array Definition
const array = ["item1", "item2"];

const array = [];
array[0] = "item1"
array[1] = "item2"

// Array Methods
array.push("new item") // adds item to array
array.pop() // removes last item
array.join() // joins array into a string
array.concat(array2) // concatinates two arrays
array.sort()
array.reverse()
array.sort(function(a,b){return a - b}) // Numerical sorting
array.forEach(someFunction) // runs a function for each value in array
array.map(someFunction) // creates a new array containing the contents of the orginal array run through some function
array.filer(value > 2) // creates a new array for all values meeting some condition
array.reduce(someFunction) // runs some function to reduce the array to a single value

function someFunction(value, index, array) {
    code
}



// Object Definition 
const objectName = {
    key1:"item1", 
    key2:"item2",
    method1: function() {// this is an object method
        return this.key1 + " " + this.key2 
    }
        // "this" means the function is refering to the current object
        // In an object, refers to the object
        // In a function, refers to the global object
        // In an event, this refers to the element that recieved the event
};

// Accessing Object Properties
objectName.propertyName
onjectName.method1()
objectName["propertyName"]
```

## Sets
Sets are arrays that have unique values, no value in a set can be repeated.
```javascript
const set = new Set(["a", "b", "c"])

// Set Methods
set.add() // add values to set
set.forEach()
set.values() // returns an iterator object containing all the values of the set
set.size() // returns number of values
set.has() // searches for value in set
set.delete() // removes value from set
```

## Maps (Similar to Python Dictionary)
```javascript
const map = new Map([
    ["key1", value1],
    ["key2", value2]
]);

const map = new Map()
map.set(["key1", value1])
map.set(["key2", value2])

// Map Methods
map.get("key1") // gets the value given a key
map.entries() // retunrs iterator with the key value pairs of the map
```