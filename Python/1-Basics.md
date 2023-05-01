# Python Basics

## Variables & Data Tyoes

```python
# Data Type Casting - Turns Data Type into another one
	type() # will return the type of function
	int() # integer (positive negative number without decimal)
	str() # string (anything embeded in quotations(' ', " "), is immutable)
	bool() # boolean (True or False) - must be capitilzaed and is a reserve word
	float() # numbers with decials
	complex() # imagenary numbers but written using 'j' intead of i	

#Varibles
	valirable = (value)
	""" 
		Naming Varibles
			- cannot contain dashes 
			-	cannont start with a number
			- cannot include spaces (use '_' instead) 
			- can only include underscorces and text
	"""

# Strings '' or "" - IMMUTABLE
	x = 'hello'
	
	# Escape Sequances
	\" #  allows you to use double quotes in a string using double quotes
	\' # single quotes
	\\ # backslash
	\n # new line
	\t # tab
	\b # backspace
	\ooo # octal value
	\xhh # hex value
	
	# Strting Concatination
	>>> print('str1' + 'str2')
			'str1str2'
	>>> print('a'*5)
			'aaaaa'
		
	# String Methods - something that can be called on by string Ex. ____.method()
	.strip() # removes leading and trailing spaces from a string
	.lower() # turns everything in the string into lowercase letters
	.upper() # turns everything in the string into uppdercase letters
	.replace(str,str, x) # replaces specific parts of string (old value, new value,count(number of times))
	.count(str,x,y) #returns number of given substrings is present in a string (x,y are start & end)
	.find() # outputs the index at the first point the perameter shows up
	.split() # creates a list out of the string given
	#Example .split()
	test = input('say somthing')
	print(text.split('.')) # will seperate the terms in the list by the location of '.' 
	#('.' wont be in list)


#Lists - []
	fruits = ['apple','pear','orange'] # indicies go from 0,1,2 (starts at 0)
	print(fruits[1]) # will output pear
	
	fruits.append('strawberry') # adds data to the list at largest position
	del fruits[1] # will deletes value at 1
	fruits[1] = 'blueberry' # will change the vale of pear to blueberry
	fruits.extend() # adds elements of a list as elements in the new list
	numbers.sort() # will sort the list acending order
	numbers.sort(reverse = True) # will sort in deciding order
	numbers.reverse() # flips the list entirely  
	numbers.remove() # removes the first occurrence of a value in the brakets from list
	numbers.count() # returns the counted number of entires of the parameter
	numbers.pop() # returns the last element from the list and removes it from the list

# Tuple - (): similar to a list but is immutable, used for data that won't be changed
	colour = (255,255,255) # different type of list used for different things like quardinates
	red, green, blue = (255, 255, 255) # tuple unpacking, you can define varibles elementwise from a tuple

# Sets - Stores unique information and does not rememebr the order
		t = set() # defines an empty set
		s = set()

		# Set methods
		set.add() # adds an element to a set
		set.clear() # removes elements from the set
		set.copy() # returns a copy of the set
		set.discard() # removes specific items
		set.intersection() # returns a set that is the intersection (common elements) between sets
		s.issubset(t) # determines if every element in s is contained in t
		s.issuperset(t) # determines if every element in t is contained in s
		s.difference(t) # returns elemts in s but not in t
		s.symmettric_difference() # returns a set with elements in either s or t but not both

		# Adding Sets together
		north_america = {'ford', 'dodge', 'tesla'}
		europe = {'BMW', 'ford', 'tesla'}

		>>> north_america.union(europe)
		{'ford', 'dodge', 'tesla', 'BMW'} # elements will only appear once in a dictionary 
		
		europe.union(north_america) # this and the following are equivelent expressions for adding dictionaries together
		europe | north_america
		north_america | europe


# Dictionaries - {}: Libraries of information acessable through keys
		months = {"Jan" : "January", "Feb" : "Feburary", "Mar" : "March", "Apr" : "April",
		"May" : "May", "Jun" : "June", "Jul" : "July", "Aug" : "August", "Sep" : "September",
		"Oct" : "October", "Nov" : "November", "Dec" : "December" }
		# Dictionaries are defined with 'Keys' that allow you to access certain elements 
		# Keys within a dictionary must be unique, there cannot be duplicates
		# Does not save the position that the element appears in the dictionary so sliciing & indexing can't be used

		# Accessing Dictionary Elements
		>> months["Mar"] # using slice operator and that key
				"March"
		>> months.get("Mar") # retrives value at a give key, returns None if not in dic
				"March"
		>> months.get("ur mum", "no you bozo") # default value when not in dictionary
				"no you bozo"

		# Iterating a dictionary
		for key in months:
				print(key) # will print each of the keys

		for key, value in months.items():
				print(key, value) # will access both the key and value of the dict

		# Dictionary methods
		dict.clear()
		dict.copy()
		dict.get() # reutrns the value of a specified key
		dict.items() # returns a view of a list containing a tuple for each key value pair (updates as dict updates))
		dict.keys() # returns a view of a list containing all keys
		dict.pop()
		dict.update() # updates dictionary with a dictionary containing any amount of key value pairs
		dict.values() # returns a view of a list that contains all the values in the dictionary
```

## Operators
```python
#Arithmetic Operators - used to perform mathematical opterations with numbers
	+ # addition
	- # subtraction
	* # multiplication
	** # exponentiation
	/ # division
	// # floor division(outputs value as integer/ignores remainder)
	% # modulus (outputs the remainder)

#Bitwise Operators - used for binary numbers
	& # "and" sets each bit to 1 if both bits are 1
	| # "or" sets each bit to 1 if one of two bits is 1
	^ # "xor" sets each bit to 1 if only one of two bits is 1
	~ # "not" inverts all bits
	<< # shift left by pushing zeros in from the right and let the leftmost bits fall off
	>> # shift right by pushing copies of the leftmost bit in the left, 
			# and let the rightmost bits fall off

#Assignment Operators - Assigns Value to a Varible
	= # assigns varible a value
	+= # equivilent to (x = x + 1) 
	-= # equivilent to (x = x - 1)
	*= # equivilent to (x = x * 1)
	**= # equivilent to (x = x ** 1)
	/= # equivilent to (x = x / 1)
	//= # equivilent to (x = x // 1)
	%= # equivilent to (x = x % 1)
	""" Bitewise Assignment - (used for bianary) """
		&= # equivilent to (x = x & 1)
		|= # equivilent to (x = x | 1)
		^= # equivilent to (x = x ^ 1)
		>>= # equivilent to (x = x >> 1)
		<<= # equivilent to (x = x << 1)

#Relational Operators - Used to compare two values
	== # equal 
	!= # not equal
	> # greater than
	< # less than
	>= # greater than or equal to
	<= # less than or equal to
""" in general these comparison operators will return booleans"""

#Logical Operators
	and # returns true if both statements are true
	or # returns true if one of the statements is true
	not() # reverse the result, returns false is statement is true

#Identity Operators
	is # returns true if both variables are the same object
	is not # returns true if both varibales are not the same object

#Membership Operators
	in # returns true if a sequence with the specified value is present in the objecy
	not in # returns true if a sequence with the specified value is not present in the object

#Slice Operator: Methid of indexing iterable data types
	fruits = fruits = ['apple','pear','orange']
	text = 'hello'
	print(text[0:3:2])# will print the letters of a string/list as if it were a for loop
	print(text[::])# leaving blank will default start at 0, complete length, and step of 1
	fruits[1:2] = 'grape'# will insert the string between the indicies in brackets
	fruits[:-1] # goes to last character but does not include last character
	
	numbers = [1,2,3]
	fruit_numbers = [fruits, numbers] # a lists of lists or a multidimensional array
	fruits_numbers[2][1] # this will call upon 2 in numbers, call the list then the indicies

# Unpack Operator - * or ** before iterable
	x = [1, 2, 234, 67, 93]
	>>> print(*x) # unpacked the list and prints the elements 
			1 2 234 67 93

	dict = {'first':1, 'second':2}
	**dict -> first=1, second=2 # ** needed to unpack dictionaries
```

## Functions
```python
#Arithmetic Operators - used to perform mathematical opterations with numbers
	+ # addition
	- # subtraction
	* # multiplication
	** # exponentiation
	/ # division
	// # floor division(outputs value as integer/ignores remainder)
	% # modulus (outputs the remainder)

#Bitwise Operators - used for binary numbers
	& # "and" sets each bit to 1 if both bits are 1
	| # "or" sets each bit to 1 if one of two bits is 1
	^ # "xor" sets each bit to 1 if only one of two bits is 1
	~ # "not" inverts all bits
	<< # shift left by pushing zeros in from the right and let the leftmost bits fall off
	>> # shift right by pushing copies of the leftmost bit in the left, 
			# and let the rightmost bits fall off

#Assignment Operators - Assigns Value to a Varible
	= # assigns varible a value
	+= # equivilent to (x = x + 1) 
	-= # equivilent to (x = x - 1)
	*= # equivilent to (x = x * 1)
	**= # equivilent to (x = x ** 1)
	/= # equivilent to (x = x / 1)
	//= # equivilent to (x = x // 1)
	%= # equivilent to (x = x % 1)
	""" Bitewise Assignment - (used for bianary) """
		&= # equivilent to (x = x & 1)
		|= # equivilent to (x = x | 1)
		^= # equivilent to (x = x ^ 1)
		>>= # equivilent to (x = x >> 1)
		<<= # equivilent to (x = x << 1)

#Relational Operators - Used to compare two values
	== # equal 
	!= # not equal
	> # greater than
	< # less than
	>= # greater than or equal to
	<= # less than or equal to
""" in general these comparison operators will return booleans"""

#Logical Operators
	and # returns true if both statements are true
	or # returns true if one of the statements is true
	not() # reverse the result, returns false is statement is true

#Identity Operators
	is # returns true if both variables are the same object
	is not # returns true if both varibales are not the same object

#Membership Operators
	in # returns true if a sequence with the specified value is present in the objecy
	not in # returns true if a sequence with the specified value is not present in the object

#Slice Operator: Methid of indexing iterable data types
	fruits = fruits = ['apple','pear','orange']
	text = 'hello'
	print(text[0:3:2])# will print the letters of a string/list as if it were a for loop
	print(text[::])# leaving blank will default start at 0, complete length, and step of 1
	fruits[1:2] = 'grape'# will insert the string between the indicies in brackets
	fruits[:-1] # goes to last character but does not include last character
	
	numbers = [1,2,3]
	fruit_numbers = [fruits, numbers] # a lists of lists or a multidimensional array
	fruits_numbers[2][1] # this will call upon 2 in numbers, call the list then the indicies

# Unpack Operator - * or ** before iterable
	x = [1, 2, 234, 67, 93]
	>>> print(*x) # unpacked the list and prints the elements 
			1 2 234 67 93

	dict = {'first':1, 'second':2}
	**dict -> first=1, second=2 # ** needed to unpack dictionaries
```

## Importing Libraries
```python
#Importing Libraries - some functions need to be imported in order to be used
import __libraryName__ # can be a defult python function or from file you created

__libraryName__.function()

from __libraryName__ import function
function() #when using from the library name isn't required when using the function 
```

## F-Strings
```python
# F Strings - allow expressions to be embedded into strings without concatination and type conversions
x = 9
>>> print(f'hello {3 + 5} there {x}')
		hello 8 there 9
```

## Conditional Statements
```python
#if statement format - first condition (must have)
if condition:
		do this
#elif statement format - middle condition(s) (can have as many as possible or none)
elif condition:
		do this
#else statement format - last condition (must have)
else condition:
		do this

#Nested conditional statements - can nest however many you want if you need to
if x > 3:
	if x == y
		print('you did it')
	else:
		print('bro you clearly suck as this')
else:
	print('YOU REALLY SUCK AT THIS GODDAM')
```

```python
# using "and"
if y == x and z > 7:
		print('are you a jokestr?')
""" both statements must be true in order for conditions to be satisfied """

# using "or"
if y == x or z > 7:
		print('are you a jokestr?')
""" one of the statements must be true in order for conditions to be satisfied """

# using not 
if not(y == x or z > 7):
		print('are you a jokestr?')
"""
the condition is satisfed if the condtion in the brackets is unsatisfied and vise versa 
"""
```

## For Loops
```python
# Syntax
for variable in iterable # variable is point in iterable, repeats for specified num of times
	body

# Iteratble Data Types
string = 'string' # iterate by each character in the string (indexing)
lists = ['string1', 'string2', 'string3'] # iterate by each set in the list
range(start, stop, step) # creates a range of numbers to iterate through


# Iterating through a string - Ex. Hello
string = 'hello'
>>> for character in hello
			print(character)
	h
	e
	l
	l
	o

# Iterating through a string by indexing
string = 'hello'
>>> for i in range(0:len(string))
			print(hello[i)
	h
	e
	l
	l
	o

#Iteratiing Through a list - Ex. Fruits
fruits = ['fruit', 'pears', 'strawberrys']
>>> for fruit in fruits 
			print(fruits[fruit]) # will print each data point in the list

	fruit
	pears
	strawberrys
```

## While Loops
```python
#Syntax
while condition == true
	do this
	break

#Example
loop = True
while loop:
	name = input("instert some shoot")
	if name == 'stop'
		break
```


## Comprehensions
```python
# Comprehensions - method of using shorter syntax to create values in a new iterable based on values in an old list

#Syntax
variable = [thing_added_to_iterable for i in range(number)]

# List Comprehensions Examples - set comprehensions work the same way
>>> x = [x for x in range(5)] # creates a for loop inside the list and adds element to the left to the list
		[0, 1, 2, 3, 4]

>>> x = [x + 5 for x in range(5)] # Ex. operations on the left side
		[5, 6, 7, 8, 9]

>>> x = [0 for x in range(5)]  # Ex. repeats with the for loop but only adds 0s
		[0, 0, 0, 0, 0]

>>> x = [[0 for x in range(3)] for x in range(3)] # Ex. matrix initiallizing 
			[[0, 0, 0],
			[0 ,0, 0],
			[0, 0, 0]]

>>> x = [i for i in range(50) if i % 5 == 0] # using if statements
		[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# Tuple Comprehensions
x = tuple(i for i in range(100)) # you have to use tuple casting because it returns a generator object otherwise (?)

#Dictionary Comprehensions
x = {i:0 for i in range(100)} # you need to assign a value because it adds based on keys 
```

## Map & Filter Functions
```python
# Map - Function that maps a function onto all the elements in a iterable
map(function, iterable) # applies the function to each element in the iterable
# map returns a map object which can be used to iterate, but cannot print (shows memory address)

# Example
nums = [1, 2, 3, 4, 5, 6]
>>> mapped_nums = list(map(lambda i: i - 1, nums))
		[0, 1, 2, 3, 4, 5]

#Filter - Function that returns a filtered object with the elements that satisfy a function condition
filter(funciton, iterable) # checks if each element statifies function condition

>>> filtered_nums = list(filter(lambda i: i % 2 ==0, nums))
		[2, 4, 6]
```

## Exceptions & Error Handling
```python
 # Exceptions - Errors or conditions that can occur
raise Exception('Bad') # forces an exception/error with a message 'Bad'

# Pass - used to prevent errors from occur within a class or something else
pass

# Try & Except - used to catch errors continue running the code if an error occurs
try:
	body # thing that you are trying to do
except type_of_error: # error not required, but any error will result in the except
	body # do this thing if the error occurs
except type_of_error as Error: # Error can be stored as a variable and printed
	print(Error) # more than one error can be used as an except
finally: # runs body of code regardless if an error occurs (optional)
	body

# Example: converting text to an integer
try:
	num = int(input('input an integer: ')) #statement does not work if an alpha string is input
except ValueError:
	print('invalid input')
 ```