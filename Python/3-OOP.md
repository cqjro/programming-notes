# Object Oriented Programming

## Magic Methods & Magic Attributes
```python
# __Magic Methods__ - Used to do specific tasks when defining classes
def __repr__(self):  # changes how the instances of a class are represented
	return f'self.__class.__.__name__('{self.value2}', ...' # example method of formatting 

# __Magic Attribures__ used to do specific things when calling an object/class
object.__dict__ # returns a dictionary of all of the attributes with in a object/class
```

## Example: Store Items
```python
# Useful Class Example: Store Items

class Item: # defines the data type and name

    # Class Attributes 
    pay_rate = 0.8
    all_items = []

    # Constructor
    def __init__(self, name: str, price: float, quantity=0): # default value sets the expected variable type	
        # Validations
        assert price >= 0, f"Price: {price} is negative"
        assert quantity >= 0, f"Quanitity: {quantity} is negative"
        
        # Assigns values to the self object attributes (Instance Attributes)
        self.name = name 
        self.price = price
        self.quanitity = quantity

    def calculate_total_price(self): # method that calcualtes the total value of that item in the store
        return self.price * self.quantity
    
    def apply_discount(self): # method that applies discount based on pay_rate,
        self.price = self.price * self.pay_rate # self.pay_rate allows for rate to be changed as an instance

    # Creating an Overall list of all objects created for this specific class (VERY USEFUL)
    Item.all_items.append(self) # appends every instance of objects created in the item class to this list

item1 = Item('Phone', 10, 10)
item2 = Item('Laptop', 20, 5)
item3 = Item('ur mom', 0, 0)

>>> for instance in Item.all_items:
    print(instance.name)
		
    Phone
    Laptop
    ur mom
```

## Creating Classes with Constructor
```python
# Classes - method of defining your own data types
	class class_name: # defines the data type and name

		# Class Attributes - Variables defined in a class
		value0 = 0.5 # these are accessable be referencing both the class or any object within the class
		class_name.value0 # (Class Level) even within the class, you must reference class attribbutes as such
		self.value0 # (Instance Level) can change this variable globally
		
		
		#Constructor Method - type of "MAGIC METHOD" that works to define Instance Attributes in a class
		def __init__(self, value1: str, value2: int, value3: bool): # sets the varibles and the types it acccepts for each
            # Run Validations to the recieved arugments
            assert value2 >= 0,  f" Value2 {value2} is negative"
            """ checks if the value meets a certain condition and returns an error with statement if not """
            
            # Assigns values to the self object attributes (Instance Attributes)
            self.value1 = value1 
            self.value2 = value2
            self.value3 = value3
			 
		# By default, self will always be an arguement for any method which referes to the specific object
		""" Notes: Anytime you define a function within a class, it is considered a method """



# Object - the actual example of the object in the class defined
example = class_name('a', 5, False) # for when an init function is defined

#Regular definition of variables in a class (effectively just whats in the init function but outside)
example = class_name()
example.value1 = 'a'
example.value2 = 5
example.value3 = False
example.urmomlol = 'ur mom lol'
""" You can use this way of defining attributes to add additional ones not in the constructor """

>>> print(example.value1) # calling the values within the object
		'a'
```

## Class Methods & Static Methods
```python
# Class Methods - Methods defined in a method only accessable from a class level

""" Used to do something with a relationship with the class, usually for instantiating objects """

# Example
import csv

class example:
		@classmethod # decorator changes the behaviour of the following function definition 
		def instantiate_from_csv(cls): # first argument for a class method is, class is first argument
            with open('items.csv', 'f') as f:
                reader = csv.DictReader(f)
                items = list(reader)
            for item in items:
                Item(
                    name=item.get('name'),
                    price=int(item.get('price')),
                    quantity=int(item.get('quantity')),
                    )

# Accessing Class Methods
example.intantiate_from_csv() # references class and not object
```
```python
# Static Methods - do not recieve instances or classes as first parameter, recieve regular inputs
""" Used for functions that are not directly needed on a instance level but are still related to the class"""
# Example for same class
	@staticmethod
	def is_integer(num):
        if isinstance(num, float):
            return num.is_integer() # returns integer of the numbers with 0 as decimals
        elif isinstance(num, int):
            return True
        else:
            return False

# Accessing Static Methods
example.is_integer(7) # called the same way as a class method but must take in a parameter like a regular function
```

## Getters & Setters
```python
# Properties - behave like attriubutes but can only be accessed or read but not changed
""" Used for cases where you want to access information but not change it"""
#Youtube Case Example

@property # this is the 'getter' portion 
def name(self):  # makes the name attribute in the class read only 
    return self.__name # suppoed to change the name of name to __name in the constructor***
""" putting "__" before an attribute makes it a private attribute not accessable outside of the class"""
    # this basically just ensures that the name thing can be defined in constructor, accessed as a property
    # but not changed since it is a read only property

@name.setter # allows you to set values for that value again
def name(self, value):
    self.__name = value
```

## Inheritance
```python 
# Inheritance - The process of creating a class that inherites the features of another
# Referencing the Youtube Reference

Item # From the example, this act as the parent class of which other classes will inherit from

class Phone(Item): # Phone is a subclass(child class) that inherits from the parent class Item
    def __init__ (self, name: str, prince: float, quantity=0, broken_phones=0)
        # "solved easier using keyworded arguements" --> look into this
        super().__init__(name, price, quantity)  # super pulls the constructor data called from parent class
				
""" super function and child classes also inherit all of the methods of the parent class """"
```

## Key Principles of Object Oriented Programming
1. **Encapsulation** - The process of restricting direct access to attributes in our class
    a. This means that you can allow methods to change these values but not directly redefining them
2. **Abstraction** - only showing necessary attributes and hiding unnecessary attributes and information from the user
    a. If methods are defined only to be used in another method, the sub methods should be inaccessible from the user
    b. private attributes and methods should be used to ensure these are only available within the class
3. **Inheritance** - Allow code to be used throughout all subclasses
    a. Ensure that methods can be used and actually using these methods in subclasses
4. **Polymorphism** - Use of a single type entity for different types of entities in different terms
    a. Ex. len() function is a single entity that knows how to handle different types of elements

Encapsulation
: The process of rescriting direct access to attributes in our class