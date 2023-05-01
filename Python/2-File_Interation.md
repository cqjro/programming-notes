# Python File Interaction

## General  Interaction
```python
#Escape Sequances
\" #  allows you to use double quotes in a string using double quotes
\' # single quotes
\\ # backslash
\n # new line
\t # tab
\b # backspace
\ooo # octal value
\xhh # hex value

# Normal Configuring File
object_name = open('path to file/name of file', 'mode ->' ("r" -> read, "w" -> write, "a" -> append, "r+" -> read & write))
object_name = open('file.txt', "r") # file.txt is the file name
object_name = open("/Users/cairo/Projects/Lab1/file.text", 'r') # pathway to the file.text stored in a Lab1 folder in the projects folder
name.close() # closes the file

#With statement Configuring File

with open('file.txt', 'r') as object_name: # confires the file to the variable object_name
		body # what we want to do with the file
# does NOT require file close statement within this method



# Reading a file
name = open('file_name.txt', 'r') # reading mode
name.readable() # returns True or False based on the mode or type of file
name.read() # returns the entire file
name.readline() # reads the first line in the file then saves the position when function is called again
name.readlines() # takes all the lines in the file and puts them into a list, indexable by line



# Writing a file
name = open('file_name.txt', 'a') # appending mode, adds text to the end of the file
name = open('file_name.txt', 'w') # writing mode, overwrites everything in the file
# if the file name in write mode doesn't exist, a new file is created
name.write("ur mom lol") #writes on the file with the inputted text
```

## CSV File Interaction
```python
# Comma Seperated Value (CSV) - Files that contained values as they would in a table set up (similar to excel set up)
# CSV Files are good ways to get information from tables into python

import csv # there is a specific module for interacting with CSV files

csv_object = open('csv_file_name.csv', 'r') # configured like any other file

reader = csv.reader(csv_object) # reader object can be used to iterate through the content of the CSV file
writer = csv.writer(csv_object) # allows you to write a csv file


# Format of Reading a CSV file - Reader outputs a lists of lists that can be iterated through
import csv

with open('csv_file_name', 'r') as csv_object:
	reader = csv.reader(csv_object)
	
	for row in reader:
			print(row)

	# Output
	['Name', 'Test1', 'Test2', 'Final']
	['John', '100', '50', '29']
	['Bob', '76', '32', '33']


# Writing to CSV - inputing a list of lists that can be iterated through
import csv

grades = [['Name', 'Test1', 'Test2', 'Final'], # the part that is being written to a csv file must be defined fist
					['John', '100', '50', '29'],
					['Bob', '76', '32', '33']]

with open('csv_file_name', 'w') as csv_object:
	writer = csv.writer(csv_object)

	for row in grades:
			writer.writerow(row)
```