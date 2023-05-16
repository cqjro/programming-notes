# Next JS
> Note: Some of these notes will be outadting for nextjs 13

Traditionally, React Apps are rendered client side. This means that the webpage is first rendered without any content. It then fetches the Javascript code for it to then render the page content. The issue with this is that search engines and social media platforms have trouble reading through this content.

Next JS is a superset of React that allows you to render the content on a server such that search engines, browsers and social media platforms are able to read/index through the fully rendered html before javascript is used for the formatting.

Next JS also stands out based on its page/routing format. Next apps are created with a pages directory, defining the different routes that the app can take, compared to the traditional ```create-react-app``` format, which emphasized a single page layout. The Next file structure will mirror the route structure that the user will naviage through. Next also has its own router to handle the routing process.

## Creating & Running a new Next JS App
```bash
cd "*insert target directory*"

npx create-next-app *insert app name*

npm run dev # this will run the app under localhost3000 in your browser
```

## CSS in Next JS
When creating a Next JS app there will be two style sheets. The first style sheet is ```globals.css``` which is a style sheet containing styles that apply for the entire app. The second style sheet is the ```Home.module.css```. This holds all of the "CSS Modules". These are used for styles that only apply to specific routes. This allows you to import the ```Home.modules.css``` and then reference the styles as javascript properties.
```javascript
// Example
import styles from '../styles/Home.module.css'

export default function Home() {
    return ( {/* This calls the style class as a javascript property within the modules file*/}
        <div className={styles.container} >    
    )
}

```

## Meta Data and Search Engine Optimization (SEO)
In order to make the app search engine optimized, the head of the html document is very important. Thus, it is important to include meta tags and titles within the head of the docuument at all times. Next JS allows you to use the ```<Head>``` tag to include this data. Additionally, there is a Next Metadata API.
```javascript
<Head> </Head>

//OR

// in the layout file
export const metadata = {
    title: 'webpage title',
    description: 'webpage description'
    // other metadata can be included here
}

```

## Next JS Pages/Routes
Next JS structures its app with a pages directory. This directory stores the pages for the app and its structure determines the route structure of the webapp. ```_app.js``` becomes the highest point in the app and is the entire point for anyone entering the app, and every individual page will start from this point.

Because of this relationship in Next, you can navigate to http://localhost:3000/fileName and it will open the page stored in this directory.

### Dynamic Routes
Dynamic routes are routes such that there will be many (or infintely many) pages that will be created under that route. This could be, for example, a shopping page for shirts with can have an infintely many number of shirts projects dynamically added to that route.

In order to achieve this relationship, a folder for for the root of said directory (Ex. shirts) will be created and contain the rest of the pages. This will contain an ```index.js``` file which will contain the webpage that lists out all of the shirts avaliable. a secondary file named ```[param].js``` (must include the square brackets), will then be used to define the dynamic route. Whenever someone routes to a component from the cars route, it will render the contents of this file dynimically with the shirt specific data.
```javascript
// [id].js

import { useRouter } from 'next/router' // to use the router

export default function Car() {
    const router = useRouter(); // defintes the router
    const { id } = router.query; // queries the router with the specific id name

    return <h1> Hello {id} </h1>
}
```

## Public Directory
This is used to store user-generated files, images and any other information that should be publically accessed and easily accessiable. 

## Root Layout
The formatting and data that is stored in the root layout can then be transfer to any children bellow in the file structure.

## Data Fetching

### Static Generation (SSG)
This renders all pages at build time. Data may become stale and hard to scale to many pages considering all of them will have to be rendered at build time.

```javascript
export async function getStaticProps({ params }) {
    // code
    // this tells next to automaically call this function on build and send the result as props to the component itself

    const res = await fetch('some url');
    const data: Repository = await res.json();
    return(
        // some jsx stuff
    )

    // this can for example, fetch data from a json file for the specific shirt based on the page id
}
```
```javascript
export async function getStaticPaths(){
    // code
    // this tells Next which dynamic pages to pre-render in advanced

    // this returns a paths object with every path in the dynamic path that needs to be rendered

    const paths = data.map(car => {
        return {params: {id: car}}
    })
}
```

### Server Side Rendering (SSR)
This generates a page at the request of the user. This is ideal for constantly changing data but is much slower and ineffcient.
```javascript
export async function getServerSideProps({ params }) {
    // does the same thing as the static generation functions but does it on every user request instead of at build times.
}
```
> Note: Both data fetching methods can be used through a next app and can be used interchangably on a per page basis.

## API Directory 
The API directory is used for setting up routes that will only apply to the server.