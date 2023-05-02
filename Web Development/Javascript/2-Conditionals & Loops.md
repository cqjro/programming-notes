

# Conditional Statements
```javascript
// If-Statement Syntax
if (condition) {
    code
} else if (condition2) {
    code
} else {
    code
}

// Swith Statement Syntax
switch(exression) { // runs expresion/function
    case value1: // if the expression is equal to value1, then this code runs
        code
        break; // stops the switch block and moves on with the rest of the script
    case value2:
        code
        break;
    case value3:
        code
        break;
    default: // if none of the cases occur, this is the default value
        // default code
}
```

# Loops
There are five different types of loops in javascript, ```for```, ```for/in```, ```for/of```, ```while```, ```do/while```.
### For Loop
```javascript
for (let i = 0, i < array.length; i++) { // defines initial value, defines max value, defines how i is incremented
    code
}

for (let expression1, expression2; expression3) { 
    code
}

// Expression1 is executed once, before the code is run
// Expression2 defines the condition that the code will run
// Expression3 is run after every time the code is run
// All expressions are optional but are only omitted in very specific situations
```

### For In Loops
```javascript
for (key in object) {
    code
}

for (value in array) {
    code
}

array.forEach(someFunction)
```

### For Of Loops
```javascript
for(variable that can be iterated) { // works for any iteratable variable (Arrays, Strings, Sets, Maps, NodeLists, etc)
    code
}
```

### While Loops
```javascript
while(condition){
    code
}

do { // this will always execute the code at least once regardless if the condition is false
    code
}
while(condition);
```

### Break & Continue
```break``` statements will cause the code to to skip the rest of the loop and continue with the rest of the script
```continute``` statements will cause the code to skip to the next iteration of the loop
```javascript
label: { // this is block label
    x + 2; // statements in label
    x + 3;
    break label; // breaks out of the specified code block
    x + 4; // will not be run
}

break labelname;
continue labelname;
```