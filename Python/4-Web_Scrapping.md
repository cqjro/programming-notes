# Web Scrapping

## Reading Website Tables
```python
import pandas as pd

# Reading the html file of a given website and returns the tables inside?
pd.read_html('website link')
```

## XPath Syntax
```python
# XPath is a language that allows you to scrape webpages

# Special Characters
/ # selects the children from the node set on the lieft side of the character
// # selects the maching node set if in the document at any level

. # referes to the present node
.. # referes to the parent node

* # selects all elements or attributes regardless of name
./* # selects all the children nodes considering the current context

@ # selects an attribute
() # groups XPath expressions
[n] # indicates taht a node with index "n" should be selected




# Basic Searching
//tagName # tag name is replaced by the actual name of the tag
//tagName[1] # selects the specific tag based on the positon

//tagName[@AttributeName="Value"] # selects an attribute with a specific attribute value

# Functions
contains() # searches for text within the tags
starts-with() # searches for tags that start with value
text() # returns the text contained within the searched element



# Example Syntax
//tagName[contains(@AttributeName, "Value")] # doesn't need exact name of value, looks for what's inside
//tagName[(expression 1) and (expression 2)] # multiple condicitons using and/or operators can be used

//article/div # returns the div element contained within the article
//article/h1/text() # returns the text within the specifed element
//article//text() # returns any text contained within an article tag
```

## Scrapping using Selenium
```python
# The following is all required in order to initizlie selenium with teh chrome driver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options # this import allows us to activate headless mode
from selenium.webdriver.chrome.service.import Service

website = '*insert website address*'
path = '*insert path to chrome driver*'

#Headless Mode
options = Options()
options.headless = True

service = Service(executable_path=path)
driver = webdriver.Chrome(service=service, options=options)
driver.get(website)

'''
body
'''

driver.quit() # when done with the program
```
```python
# Selenium Methods

# by idicates how the value is located (XPath), value is the value of the method of search (the actual XPath)
driver.find_element(by='', value='') 
driver.find_elements(by, value) # returns all the values contained within the XPath specified

xpath.get_attribute('*insert attribute name*')


# Method of Iterating through similar container elements

containers = driver.find_elements(by="xpath", value='//div[condition]') # get all of the containers

for container in containers:
	container.find_element(by='xpath', value='./a/h2') #. is the current element(container), returns the rest in each container
```
