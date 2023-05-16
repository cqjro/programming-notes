# React JS
> From the [Oden Project React Module](https://www.theodinproject.com/lessons/node-path-javascript-react-introduction)

React is a library that can be imported into the current code. Additionaly, React focuses on component based development. This causes projects to be written in seperate component files which are then subsequently importented in to the main ```App``` script, which is the root of the web app. 

## Creating a React App
```bash
npx create-react-app my-first-react-app # creates default template for a React App
npm start # this will open the project in your browser, or you can open in text editor
```
### Index.js and App.js
Index,js is the default entry point of the application. It contains the following code which will load the the App component into the DOM, specifically the element with the ```root``` id. 
```javascript
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

---
## Components

### React Class Components
```javascript
import React, { Component } from 'react' 
// importing react and component from the react library
// Default exports are not wrapped in curly brackets, but other exports must be

class App extends Component { 
// creating a componenet class, and giving it all the methods and properties that a react component has
    constructor() {
        super() // must be written for any react constructors
    }

    {/* Javascript functions can be written here */} // weird comment business

    render() {
        // this is written in JSX, which is similar to HTML, but is converted into JavsScript such that a browser is able to process it.
        // All React Component Classes need a render function
        return (
            <div className="App"> 
                Hello World!
            </div>
        )
        // Render function is a lifescycle function
        // the Render function only returns the top level element contained within it and thus, any other elements must be nested within the first one
    }
}

export default App // creates the default export for the App Component Class
```

### React Functional Components
```javascript 
import React from 'react';

function App() {
  return <div className="App">Hello World!</div>;
}

// OR (arrow-function syntax)

const App = () => {
  return <div className="App">Hello World!</div>;
};

// OR (implicit return)

const App = () => <div className="App">Hello World!</div>;

export default App;
```
---
## Props
 ```props``` are the way that components share values and functionality. This effecitvely means through inheritance, you can pass down both properties of a class and functions/methods of a class to the various components.
 
 ### React Classes
 This is done by passing the props argument into the ```constructor``` and ```super()``` functions within the **child component** class definition. This is what allows the propeties to be defined and thus transfered when defined as a child component. 
 
 ```super()``` is used in call functions or look up properties that are contained in the "super" class aka the parent class. It must be initialized with all React classes, and must be used anytime the defined class is inheriting from a parent class.
 ```javascript
super([arguments]) // calls the parent constructor.
super.propertyOnParent
super[expression]
 ```

 ### Passing down Properties
 Defining properties in the App.js script can then pass down the properties of the defined components to those components so that they can then be used.
```javascript
// The MyComponent.js file

import React, { Component } from 'react';

class MyComponent extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    // renders the title property of the component
    return (
      <div>
        <h1>{this.props.title}</h1> 
      </div>
    );
  }
}

export default MyComponent;
```

```javascript
// The App.js file

import React, { Component } from 'react';
import MyComponent from './MyComponent';

class App extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    // here, MyComponent is rendered as a child component of App
    // the component has the title property defined within the app
    return (
      <div>
        <MyComponent title="React" /> 
      </div>
    );
  }
}

export default App;
```

### Passing down Functions
This works the exact same way as properties. Functions are defined in the ```App``` class and then are passed down to the child component classes.
```javascript
// The MyComponent.js file

import React, { Component } from 'react';

class MyComponent extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    // Event listeners in react are defined directly in the JSX 
    // This is instead of added using the addEventListener method
    // The javascript mused be added using {} when placed in JSX code
    return (
      <div>
        <h1>{this.props.title}</h1>
        <button onClick={this.props.onButtonClicked}>Click Me!</button>
      </div>
    );
  }
}

export default MyComponent;
```
```javascript
//The App.js file

import React, { Component } from 'react';
import MyComponent from './MyComponent';

class App extends Component {
  constructor(props) {
    super(props);

    this.onClickBtn = this.onClickBtn.bind(this);
    // this binds the method to this, which must be done when passing functions between class components
  }

// this defines the respective method for the button click
  onClickBtn() {
    console.log('Button has been clicked!');
  }

  render() {
    return (
      <div>
        <MyComponent title="React" onButtonClicked={this.onClickBtn} />
      </div>
    );
  }
}

export default App;
```

### Destructing
Destructing is a way in which we can clean up long and repeated prop calls.
```javascript
// MyComponent.js

import React, { Component } from 'react';

class MyComponent extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    const { title, onButtonClicked } = this.props;
    // This deconstructs the title and onButtomClicked from this.props 
    // By Deconstructing them, they can now be refered by just their names
    return (
      <div>
        <h1>{title}</h1>
        <button onClick={onButtonClicked}>Click Me!</button>
      </div>
    );
  }
}

export default MyComponent;
```
## State
States are ways to handle values that change overtime, basically tracking the "state" of something. The state holds the values that change overtime. States should be treated as immutable and never changed directly such as using ```this.state.count = 5``` as this can cause errors. States should only be changed using the ```setState()``` method.
```javascript
import React, { Component } from 'react';

class App extends Component {
  constructor() {
    super();

    this.state = { // sets the state properties similar to props
      count: 0,
    };

    this.countUp = this.countUp.bind(this); // binds function to this
  }

  countUp() { // defines the function to change the state
    this.setState({ // the state can only be changed using the set state
      count: this.state.count + 1, // this can be destructered
    });
  }

  render() {
    return (
      <div>
        <button onClick={this.countUp}>Click Me!</button>
        <p>{this.state.count}</p>
      </div>
    );
  }
}
```

### Passing State as Props
The ```state``` of any component can be passed down to any other component through ```prop``` such that child components can be re-rendered with the new state values. This is best done by passing the ```state``` as a ```prop``` within the App.js render method.
```javascript
// in the render method of App.js
return (
  <div>
    <NavBar username={this.state.username} />
    <Forum username={this.state.username} />
    <Footer />
  </div>
);
```

## State & Props in Functional Components
Functional components are often perfered by React developers. They allow for state setting and access through something called **React Hooks**.

The main difference between the functional and class based components when it comes to ```props``` and ```state``` is that ```props``` is passed as an argument to the function instead of in the ```constructor()``` and ```super()```. ```props``` are also accessed through the means of ```props.functionName``` instead of ```this.props.functionName```.

```javascript
// MyComponent.js

import React from 'react';

function MyComponent(props) {
  return <div>{props.title}</div>; // this can be destructured
}

export default MyComponent;
```

```javascript
// App.js

import React from 'react';
import MyComponent from './MyComponent';

function App() {
  return (
    <div>
      <MyComponent title="Hello World" />
    </div>
  );
}

export default App;
```

### Functional Destructing
Does not require the same destructure line and can be done within the arguement passed in the functional definition.
```javascript
const {title} = props // destructure line

function MyComponent({ title }) { // this method does not require the first line
  // rest of code
}
```
---
## Lifecycle Methods
Lifecycle methods are the React Methods that operate React Components on the DOM. These are the methods that can only be used within class components. These are used to create, render, update and remove components.

```componentDidMount()```
: This method is run when inserting the component into the DOM. This is used for APIs or other frameworks, timers and event listeners

```render()```
: This method is placed in React class components in order to display the component content on the webpage. Additionally, rendering can be conditional with relevant logic specified for its rendering.

```ComponentDidUpdate()```
: This is used whenever the component updates. It is prone to infinite loops. but is useful for updating the component content.

```ComponentWillUnmount()```
: This is used when the component is removed from the DOM, usually as a clean up task.

---

## Hooks

These are the function equivilent to lifecycle methods in class components and also allow functions to access ```state```.

### ```setState()```
This allows you to set the state within a functional component.
```javascript
import React, { useState } from "react"; // importing the state hook

const App = () => {
  const [count, setCount] = useState(0); // initializes the state with value of 0

  const incrementCount = () => {
    setCount(count + 1); // changing the state causes a re-render on the webpage
  };

  return (
    <div>
      <div>{count}</div>
      <button onClick={incrementCount}>Increment</button>
    </div>
  );
};

export default App;
```

### ```useEffect```
This replaces the lifecycle methods found in class components. It will change the display of the componet anytime a variable left in the dependency array is changed, allowing for continous updates.

```javascript
// Syntax
useEffect((args) => {code that will be executed}, [optional: dependancy array])
```
```javascript
import React, { useState, useEffect } from "react";

const App = () => {
  const [color, setColor] = useState("black");

  useEffect(() => {
    const changeColorOnClick = () => {
      if (color === "black") {
        setColor("red");
      } else {
        setColor("black");
      }
    };
    
    document.addEventListener("click", changeColorOnClick);

    return () => {
      document.removeEventListener("click", changeColorOnClick); 
      // this return statement makes the useEffect equivilent to componentWillUnmount 
    };
  }, [color]);

  return (
    <div>
      <div
        id="myDiv"
        style={{
          color: "white",
          width: "100px",
          height: "100px",
          position: "absolute",
          left: "50%",
          top: "50%",
          backgroundColor: color,
        }}
      >
        This div can change color. Click on me!
      </div>
    </div>
  );
};

export default App;
```

---

## Router