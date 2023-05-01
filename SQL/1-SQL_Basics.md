# SQL Basics

## Selecting Data
```sql
--In SQL, Data is read as if they were in tables

SELECT property_name --This is a querey
FROM table_name;

SELECT * -- Selects all of the columns of the table
FROM table_name;

SELECT *
FROM table_name
WHERE column_name = 'property' -- selects rows with the specified conditions

SELECT DISTINCT * -- Selects non-duplicate values from the specificed columns
FROM table_name


-- Types of WHERE conditions
WHERE column_name = 1
WHERE column_name <> 1 -- not equal to 1
WHERE column_name > 0
WHERE column_name <= 0
WHERE column_name BETWEEN 1970 AND 2000
WHERE column_name LIKE 's%' -- Returns datta that starts with s
WHERE column_name LIKE '%s%' -- data that contains an s in it
-- the % represents an unknown continuation
WHERE column_name IN (item1, item2) -- Returns data equal to the items within the list
-- ^ in statement can also be used for sub queries (queries in brackets)

-- Boolean Operators
AND
OR
NOT


-- Limits
LIMIT number -- Limits the number of rows displayed, used for large databases

-- Aliasing
SELECT column_name AS alias_name -- changes the name of the COLUMN in the QUERY ONLY
FROM table_name;

SELECT column_name(s) -- changes hte name of the TABLE in the QUERY ONLY
FROM table_name AS alias_name;

-- Union of Selections: Combines the data from both tables into one view
-- Note: Must have the same number of rows and UNION ignore duplicates unless using UNION ALL
SELECT column_name(s) FROM table1
UNION ALL
SELECT column_name(s) FROM table2;

-- Grouping Data: groups the data types in certain column, 
-- usually pairs with functions to determine certain information
SELECT COUNT(column_name), column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
ORDER BY column_name(s);

-- Having: Allows aggrate functions to be used within a query
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country
HAVING COUNT(CustomerID) > 5;

-- Exists: Tests for the existance of data
SELECT SupplierName
FROM Suppliers
WHERE EXISTS 
(SELECT ProductName 
	FROM Products 
	WHERE Products.SupplierID = Suppliers.supplierID AND Price < 20
);
```

## Ordering Data
```sql
ORDER BY column_1 -- orders the data based on the specific column values, aecending default
ORDER BY column_1, column_2 ASC -- change to acending
ORDER BY column_1, column_2 DESC -- change to decending

-- the multiple columns means that it will first order by column_1, 
-- then column_2 if the values of column_1 have the same value
```

## Inserting Data
```sql
INSERT INTO table_name (column1, column2, column3, ...)
VALUES (value1, value2, value3, ...);

-- when using auot increment for the primary key, that value can be omitted
-- if included, auto increment will be over written
```

## Updating Data
```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;

-- used to update specfic data points
```

## Deleting Data
```sql
DELETE FROM table_name
WHERE condition;

-- WARNING: Not including the WHERE clause will result in the entire table being deleted
```

## Aggregate Functions
```sql
SELECT MIN(column_name) -- returns the lowest number in column
FROM table_name
WHERE condition;

SELECT MAX(column_name) -- returns the highest number in column
FROM table_name
WHERE condition;

SELECT COUNT(column_name) -- counts the number in the column
FROM table_name
WHERE condition;

SELECT AVG(column_name) -- averages the values in the column
FROM table_name
WHERE condition;

SELECT SUM(column_name) -- sums the values in a column
FROM table_name
WHERE condition;
```

## Joins
```sql
SELECT MIN(column_name) -- returns the lowest number in column
FROM table_name
WHERE condition;

SELECT MAX(column_name) -- returns the highest number in column
FROM table_name
WHERE condition;

SELECT COUNT(column_name) -- counts the number in the column
FROM table_name
WHERE condition;

SELECT AVG(column_name) -- averages the values in the column
FROM table_name
WHERE condition;

SELECT SUM(column_name) -- sums the values in a column
FROM table_name
WHERE condition;
```

## Other
```sql
--FIXING THE AUTO INCREMENT
ALTER TABLE exercise_library AUTO_INCREMENT = 4;

--IMPORTING FROM A CSV FILE
--@block 
LOAD DATA LOCAL INFILE '/Users/cairo/dev/SQL Projects/Fitness Database/exercise_library_insert.csv'
INTO TABLE exercise_library
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
```