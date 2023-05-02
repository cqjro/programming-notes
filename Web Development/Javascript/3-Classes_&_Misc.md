> Based on the [W3Schools Javascript Tutorial](https://www.w3schools.com/js/default.asp)

# Classes
```javascript
class ClassName {
    constructor(input1, input2) {
        this.name = input1
        this.other = input2
    }
    method1(arg1) {
        code
    }
}
```

# Miscellaneous Stuff

## JavaScript Regular Expressions
Regular expressions are sequences of characters that form a search pattern, which can either be used to simiply search for text or replace text.
```javascript
// Syntax
/search/modifiers // modifiers how the search is conducted (Ex. Case sensitive)
// These are commonly used in string methods such as search() or replace()

// Modifiers
/searchword/i // case sensitive search
/searchword/g // global match (finds all matches instead of just one)
/searchword/m // multiline matching

// Pattern Expressions
/[abcd]/ // finds any of the characters inside the brackets
/[0-9]/ // fines any of the didgets within the brackets
/(a|b|c)/ // find any of the alternatives sperated by |
```

## Error Handling 
```javascript
try {
    code
}
catch(error) {
    code
}
finally { // run regardless of the try and catch
    code
}

throw custom error
```

## Modules
Modules allow you to break up code into sepearte files and import them. This only work with HTTP(s) protocols and not file:// protocol.
```html
<script type="module">
    import message from "./message.js/"
</script>
```
### Named Exports
```javascript
export const variable = "hello";
// or

const variable1 = value1
const variable2 = value2

export {variable1, variable2}; // can be placed at the bottom of the script
```

### Default Exports
You can only have one default export per file.
```javascript
const message () => { // this is goofy function notation
    const name = "hello"
    const variable = 2
};

export default message; // exports this by default
```

## JSON
JSON files use javascript object notation as a lightweight method of storing data. These can be read using any programming laguage despite using the javascript object syntax.
```json
"key":"value" // JSON data is defined using key:value pairs

"employees":[ //JSON can define arrays of objects
  {"firstName":"John", "lastName":"Doe"},
  {"firstName":"Anna", "lastName":"Smith"},
  {"firstName":"Peter", "lastName":"Jones"}
]
```
### JSON Text to JavaScript Object
```javascript
let text = '{ "employees" : [' +
'{ "firstName":"John" , "lastName":"Doe" },' +
'{ "firstName":"Anna" , "lastName":"Smith" },' +
'{ "firstName":"Peter" , "lastName":"Jones" } ]}';

const obj = JSON.parse(text); // converts the JSON text into a javascript object
```