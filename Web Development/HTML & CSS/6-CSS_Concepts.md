# Important CSS Concepts
>  From the [Odin Project Intermediate HTML & CSS](https://www.theodinproject.com/paths/full-stack-javascript/courses/intermediate-html-and-css) 

---
## Selectors
### Parent & Sibling Combinators
Selectors are used to refer to specific children when refering to a parent. This is used for things like refering to the concents of a flex box container.
```css
> /* child comnbinator */
+ /* adjacent sibling combinator */
~ /* general sibling combinator */
```
### Pseudo-Selectors
These are selectors that reference items in the HTML document that don't normally exist in the HTML document.

#### Pseudo-Classes
```css
/* Events */
:focus /* applies to element in focus my mouse or keyboard */
:hover /* applies to element being hovered over */
:active /* apples to element currently being clicked */
:link
:visited /* visted links */

/* Structural */
:root /* represents the top level of the document, used for global rules */
:first-child
:last-child
:empty /* elements with no children */
:only-child /* elements with no siblings */
:nth-child /* used to select elements in a pattern such as even elements, or for lists */
```
#### Pseudo-Elements
```css
::marker /* bullet point styling */
::first-letter
::first-line
::selection /* changes the user highlighting on the screen */
::before /* adds elements to the page using CSS instead of html */
::after /* adding elements with CSS is typically used to decorate text */
```

### Attribute Selectors
```css
[attribute] /* selects anything where this attribute exists */
selector[attribute] /* this links any selector (class, element, pseudo) with the attribute selector */
[attribute="value"] /* this allows you to specify what the attribute you are looking for is equal to */

[attribute^="value"] /* ^= Will match strings from the start. */
[attribute$="value"] /* $= Will match strings from the end. */
[attribute*="value"] /* *= The wildcard selector will match anywhere */

/* this allows you to select for example, classes with similar starting or ending names together */

```
---

## Positioning 

### Statics & Relative Positioning
```css
.class {
  position: static; /* default positioning of every element */
  position: relative; /* same as static but other properties allow change in location */
  position: absolute; /* allows you to position things to an exact point without distrupting other elements */
  top: 8rem;
  bottom: 7rem; /* these are used to position the elements */
  right: 4rem;
  left: 1rem;
}
```

### Fixed & Sticky Positioning
Fixed elements are positioned releative to the view port. 
Sticky elements behave like normal elements, until you scroll past them and then they behave like fixed elements.

---


## Custom Properties (CSS Variables)
```css

.error-modal {
  --color-error-text: red; /* This is how you declate custom properties*/

   color: var(--color-error-text); /* this is how you call custom properties */
  color: var(--color-error-text, white); /* the second value is a fallback value incase the first value is invalid or undefined */
}
```

### Custom Property Scope
Custom properties are limited to the scope of the selector they are defined in. This means that only the element selected **AND ITS CHILD ELEMENTS** will be able to access the variable. Global variables must be defined in the ```:root``` selector.
```css
:root {
  --main-color: red; /* all elements can access this variable */
}
```

# THEMES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
> [Odin Project Themes Example](https://www.theodinproject.com/lessons/node-path-intermediate-html-and-css-custom-properties) is sick as a reference.

Basically just using the custom properties to indicate the parts of the website that are changing and use javascript to change these properties based on some user event (button press).

Also, light and dark mode can be selected automatically by using a media query ```prefers-color-scheme``` which takes information from the OS/browser to set the theme based on whatever the user has their preference set to. This only works for light and dark mode however, and other handling must be applied for alternative themes.

---

## CSS Functions
### Colour Functions
```css
.class {
  color: rgb(0, 42, 255); /* takes in the rgb values as input and calculates the exact colour */
  background: linear-gradient(90deg, blue, red); /* takes in two colour argumnets and angle argument to create a gradient of colours */
}
```

### Number Functions
```css
/* calc() Function */
:root {
--header: 3rem; /* CSS variables */
--footer: 40px;
--main: calc(100vh - calc(var(--header) + var(--footer)));
/* able to calculate output even with mix matched units */
/* able to be nested for more complex operations */
}

/* min() & max() */
#iconHolder {
  width: min(150px, 100%); 
  /* will always output the minimum of the two input values */
  /* in this case, since 150px is smaller, then the width will be 150px 
  however, if 150 px is not avaliable, it will output 100% as that is now the smaller option*/

  height: min(150px, 100%);
  box-sizing: border-box;
  border: 6px solid blue;
}

.class{
  width: max(100px, 4em, 50%);
  /* operates the same as min just with the largest value */
}

/* clamp() Function */
h1 {
  font-size: clamp(320px, 80vw, 60rem);
  /* inputs: smallest value, ideal value, largest value*/
  /* makes the styling adaptive and sets a minimum, maximum and ideal case based on avaliability */
}
```
---

## Flex Box 
### Flex Blox Basic Syntax 
```html
<!-- Containers are block elements and take up the full width of the page -->
<div class="flex-container"> 
  <div class="one"></div> <!-- Flex items: can be containers themselves with their own items -->
  <div class="two"></div>
  <div class="three"></div>
</div>
```
```css
.flex-container {
    display: flex; /* this automatically displays the div elements horitzontally instead of veritically */
}

/* this selector selects all divs inside of .flex-container */
.flex-container > div {
  background: peachpuff;
  border: 4px solid brown;
  height: 100px;
  flex: 1; 
}

.flex-container > .one { /* must be included because it div takes priority */
  background: peachpuff;
  border: 4px solid brown;
  height: 100px;
  flex: 2; /* grows twice the size of the other divs within the container */
}

```


### Flex Shorthand
Shorthand properties in CSS are effectively short forms of properties that allow to you set the values of mutiple CSS properaties at the same time.

The ```flex``` declaration is shorthand for ```flex-grow```, ```flex-shrink``` and ```flex-basis```.

The ```flex-grow``` property determines how much the flex item will grow realative to its sibling items.
The ```flex-shrink``` items determins the rate at which shrinking occurs when bellow the basis size.

```css
div {
    flex: 1; 
    /*  this means flex-grow: 1, flex-shrink: 1, flex-basis: 0 */
    /* These values are relative to the other values within the container */
    /* Basically means it'll change size based on the size of the container */
    /* flex: 2 willl also change but have double the length of flex: 1*/

    flex g s b; /* specifiying the grow, shrink, and basis*/

    flex 2 1 auto; /* auto basis makes it so that it has to check for a specified width */
}
```

### Flex Axes
Flex boxes can work in both the horizontal (row) or the vertical (column) direction, with certain rules applying in either direction.
```css
.flex-container {
  flex-direction: column; /* changes the main flex axis */
}
```

### Alignment
Content will expand to fill all avaliable space when used in a flex box, however, specifiying the alignment will instead arrange the items more effectively.
```css
.container {
  display: flex;
  justify-content: space-between; /* justifies on the specified axes with even spaces between items */
  justify-content: center;
  
  align-items: center; /* Aligns items along the alternate axis */

  gap 8px; /* specifies the gap between items */
}
```
---

## Grid
It is possible to use flex boxes to display elements in a grid style using ```flex-wrap``` however, the grid method makes this a lot smoother.
```css
.grid-container {
  display: grid; 
  /* established element as a grid container, can also use grind-inline */
  /* direct child elements become grid elements automatically */
  /* grandchild elements do not become grid elements, only direct children*/

  grid-template-columns: 75px 75px 75px;
  grid-template-rows: 75px 75px;
  /* these define the column and row tracks of the grid */

  grid-template: 50px 50px / 50px 50px 50px;
  /* this is shortahnd for the columns and rows template properties */

  grid-auto-rows: 50px;
  grid-auto-columns: 50px;
  /* this will implicitly set the size of new columns or rows added 
  that are not specified in the original template */

  grid-auto-flow: column; 
  /* changes the way that items are automatically added 
  to the grid from new row addition to new column addition */

  column-gap: 20px;
  row-gap: 50px;
  gap: 32px; 
  /* shorthard for both the row and column properties*/
}
```

### Positioning
```css
#id {
  grid-column-start: 1;
  grid-column-end: 6;
  grid-column: 1/6; /* short hand */
  /* defines the span of a single grid item for columns */

  grid-row-start: 1;
  grid-row-end: 3;
  grid-row:  1/3; /* short hand */
  /* defines the span of a single grid item for rows */

  grid-area: 1/1/3/6;
  /* short hand for row-start/column-start/row-end/column-end */
}
```

#### Grid Area Templates
```css
.container {
  display: inline-grid;
  grid-template: 40px 40px 40px 40px 40px / 40px 40px 40px 40px 40px;
  background-color: lightblue; 
  grid-template-areas:
    "living-room living-room living-room living-room living-room"
    "living-room living-room living-room living-room living-room"
    "bedroom bedroom bathroom kitchen kitchen"
    "bedroom bedroom bathroom kitchen kitchen"
    "closet closet . . ." 
    /* . is used to specifity an empty cell */    
}

.room {
  border: 1px solid;
  font-size: 50%;
  text-align: center;
}

#living-room {
   grid-area:  living-room; 
   /* you can assign the grid area to the area specified in the template by a specific string */
}

#kitchen {
  grid-area: kitchen;
}

#bedroom {
  grid-area: bedroom;
}

#bathroom {
  grid-area: bathroom;
}

#closet {
  grid-area: closet;
}
```

### Advanced Grid Properties
```css
.grid-container {
  display: grid;
  resize: both; /* allows the container to be resized from the bottom corner */
  overflow: auto; /* allows scrolling if grid-container is smaller than grid */
  padding: 1px;
  gap: 1px;

  grid-template: repeat(2, 150px) / repeat(5. 150px);
  /* repeat() allows for repetition of information for defining the template */

  grid-template: repeat(2, 1fr) / repeat(5. 1fr);
  /* the fr unit gives a relative fraction of the avaliable space to the cell */

  grid-template: repeat(2, min(200px, 50%)) / repeat(5. min(120px, 15%));
  /* using min (or max) can allow for appropriate sizing for different browser window sizing */

  grid-template: repeat(2, minmax(200px, 300px));
  /* sets a static value for the minimum and maxium, only works for grids */

  /* Note: Clamp also works, with the percentage used as the ideal case */

  grid-template: repeat(auto-fit, 200px);
  /* will always return the largest possible integer value without overflowing the container */


}


```
