# ============================================================
# Seminar 1: Python Fundamentals for Business
# Goals:
#   - Get comfortable reading and writing Python code for business contexts
#   - Build basic “business logic” using variables, calculations
#   - Practise core programming structures: types, conditions, loops, lists, dicts, functions
#   - Learn NumPy and Pands fundamentals for efficient numerical computing
# ============================================================

# ============================================================
# Warming up: print()
# ============================================================

# The print() function is the simplest way to display information.
# In practical work, print() is also a debugging tool: if you are unsure what a variable contains,
# you can print it to check quickly.

print("Hello, MSc students!")
print("Welcome to Python for Business.")


# Explanation:
# Each print() writes a new line by default, so you see two separate lines.
#
# Common mistakes:
# - Forgetting quotation marks around text (Python then treats the word as a variable name).
# - Expecting print() to “return” a value (print displays text but returns None).


# ============================================================
# Basic arithmetic and core maths operations
# ============================================================

# This section introduces Python’s arithmetic operators and how they behave with integers and floats.
# You should pay attention to the difference between / (true division) and // (floor division).

a = 10
b = 3

print("Basic arithmetic:")
print("a + b =", a + b)    # 13
print("a - b =", a - b)    # 7
print("a * b =", a * b)    # 30
print("a / b =", a / b)    # 3.333...
print("a // b =", a // b)  # 3
print("a % b =", a % b)    # 1
print("a ** b =", a**b)    # 1000


# Explanation:
# - a / b produces a float because true division keeps decimals.
# - a // b removes the decimal part and keeps the floor value.
# - a % b returns the remainder after division.
# - a ** b is exponentiation.
#
# Common mistakes:
# - Using ^ for power (in Python, ** is power; ^ is bitwise XOR and gives a different result).
# - Confusing // with / in financial calculations (floor division can silently destroy decimals).


# ============================================================
# More maths with built-ins and the math library
# ============================================================

# Python can do more advanced maths using built-ins (abs, round, etc.) and standard libraries.
# So we need to import the math library into our environment.

import math

print("\nMore maths:")
print("Square of a (a^2) =", a**2)
print("Square root of a =", math.sqrt(a))
print("Absolute value of -7 =", abs(-7))


# Explanation:
# "\n" creates a blank line before the text, which improves readability in console output.
#
# Common mistakes:
# - Forgetting to import math before calling math.sqrt().


# ============================================================
# Variables and basic types
# ============================================================

# Python variables store values, and each value has a type that affects what you can do with it.
# In business analytics, type awareness matters because:
# - revenue is usually a float (currency)
# - order counts are usually integers
# - flags (active/inactive) are booleans

company = "SR Mailing LTD"   # string
week = 1                     # integer
revenue = 125000.50          # float
active = True                # boolean

print("\nVariables and types:")
print(company, week, revenue, active)
print(type(company), type(week), type(revenue), type(active))

# Expected output (example):
# Variables and types:
# SR Mailing LTD 1 125000.5 True
# <class 'str'> <class 'int'> <class 'float'> <class 'bool'>
#
# Explanation:
# Printing multiple items separated by commas displays them with spaces between.
#
# Common mistakes:
# - Storing a number as text (e.g., revenue = "125000.50") which breaks arithmetic later.
# - Using "True" (string) instead of True (boolean).


# ============================================================
# Arithmetic and business metrics
# ============================================================

# This section shows a typical analytics pattern: computing a business metric from variables.
# Average order value (AOV) is defined as revenue divided by the number of orders.

orders = 320
avg_order_value = revenue / orders

print("\nBusiness metrics:")
print("Average order value:", avg_order_value)
print("Average order value (rounded):", round(avg_order_value, 2))

# Expected output (example):
# Business metrics:
# Average order value: 390.6265625
# Average order value (rounded): 390.63
#
# Explanation:
# Currency is usually displayed to two decimal places for reporting clarity.
#
# Common mistakes:
# - Using // instead of /, which removes decimals and causes reporting errors.
# - Not considering division by zero when orders could be 0.


# ============================================================
# Comparisons, boolean logic, and decision rules
# ============================================================

# Comparisons return booleans (True/False), and booleans are the foundation of decision rules.
# Many business rules depend on thresholds and conditions (targets, risk limits, inventory triggers, etc.).


# ------------------------------------------------------------
# Comparisons return True/False
# ------------------------------------------------------------

# This part introduces comparison operators: ==, !=, <, <=, >, >=.

a = 10
b = 12

print("\nComparisons return booleans (True/False):")
print("a < b:", a < b)
print("a <= b:", a <= b)
print("a > b:", a > b)
print("a >= b:", a >= b)
print("a == b:", a == b)
print("a != b:", a != b)


# Common mistakes:
# - Confusing == (comparison) with = (assignment).
# - Accidentally writing a = b inside a condition (this is invalid in Python, but the idea matters).


# ------------------------------------------------------------
# Boolean operators: and / or / not
# ------------------------------------------------------------

# Boolean operators let us combine conditions into more realistic business logic.

high_revenue = revenue >= 100000
high_orders = orders >= 300

print("\nBoolean operators:")
print("high_revenue:", high_revenue)
print("high_orders:", high_orders)
print("high_revenue AND high_orders:", high_revenue and high_orders)
print("high_revenue OR high_orders:", high_revenue or high_orders)
print("NOT high_revenue:", not high_revenue)

# Explanation:
# - and requires both conditions to be True
# - or requires at least one condition to be True
# - not flips True to False (or False to True)


# ------------------------------------------------------------
# Decision rule with multiple branches (if / elif / else)
# ------------------------------------------------------------

# A business rule often needs multiple categories (not just pass/fail).
# The order of conditions matters because Python checks from top to bottom and stops at the first match.

target = 100000

print("\nDecision rule with multiple branches:")
print("Revenue:", revenue, "Target:", target)

if revenue >= target * 1.20:
    print("Excellent: significantly above target.")
elif revenue >= target:
    print("On target or slightly above.")
else:
    print("Below target.")

# Common mistake:
# If you check the weaker condition first (revenue >= target),
# the stronger condition (revenue >= target * 1.20) might never run.


# ------------------------------------------------------------
# Very important: indentation matters
# ------------------------------------------------------------

# Indentation is part of Python’s syntax and determines which lines belong to a block.
# Incorrect indentation can cause an error, or worse, produce wrong logic without obvious errors.

x_demo = 5
if x_demo < 3:
    print("Correct: this line is inside the if-block (because it is indented).")
print("This line is outside the if-block (no indentation).")

# Expected output (example):
# This line is outside the if-block (no indentation).
#
# Explanation:
# The second print runs regardless of the if-condition because it is not indented.


# ============================================================
# Lists
# ============================================================

# Lists store an ordered collection of items.
# In business analytics, lists often represent repeated values such as products, weekly sales, or customer IDs.

products = ["Box", "Mailer", "Tape", "Label"]
costs = [0.40, 0.25, 1.20, 0.05]

print("\nProducts:", products)
print("First product:", products[0])
print("Number of products:", len(products))

# Notes:
# - Indexing starts at 0, not 1.
# - Using an index that is out of range causes an IndexError.


# ============================================================
# Loops
# ============================================================

# for-loops let you repeat actions across items.
# Accumulation (start at 0, then add each value) is a very common business programming pattern.

for p in products:
    print("Product:", p)

total_cost = 0.0
for c in costs:
    total_cost = total_cost + c

print("Total cost:", total_cost)


# Explanation:
# Floating-point values can show small precision artifacts (e.g., 1.9000000000000001).
#
# Common mistakes:
# - Forgetting to initialise total_cost before the loop.


# ============================================================
# Dictionaries
# ============================================================

# Dictionaries store key-value pairs.
# They are common in business analytics because they map identifiers to values
# (e.g., product -> unit cost, country -> tax rate, segment -> discount level).

price = {"Box": 0.40, "Mailer": 0.25, "Tape": 1.20, "Label": 0.05}

print("\nPrice of Tape:", price["Tape"])

price["Bubble wrap"] = 0.30
print("Updated price dictionary:", price)

# Common mistakes:
# - Accessing a key that does not exist causes a KeyError.
# - Confusing dictionary keys with list indices (they are different concepts).


# ============================================================
# Functions
# ============================================================

# Functions package logic into reusable components.
# In analytics, this helps you apply rules consistently across many data points.

def classify_order(value):
    if value >= 150:
        return "Large"
    elif value >= 60:
        return "Medium"
    else:
        return "Small"

order_values = [120, 35, 78, 210, 55]

print("\nOrder size classification:")
for v in order_values:
    print(v, "->", classify_order(v))

# Common mistake:
# Forgetting to return a value leads to None, which breaks later computations.


# ============================================================
# NumPy: array operations
# ============================================================

# NumPy is the standard library for numerical computing in Python.
# NumPy arrays support vectorised operations, meaning arithmetic can be applied to entire arrays efficiently.

import numpy as np

# ------------------------------------------------------------
# Creating NumPy arrays
# ------------------------------------------------------------

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("\na:", a)
print("b:", b)
print("Shape of a:", a.shape)

# Explanation:
# For a 1D array with four elements, shape is (4,).
#
# Common mistake:
# Confusing (4,) with (4, 1), which is a 2D column vector.


# ------------------------------------------------------------
# Vectorised operations
# ------------------------------------------------------------

print("\nVectorised operations:")
print("a + b:", a + b)
print("a * 2:", a * 2)
print("a * b:", a * b)

# Explanation:
# These operations happen element-by-element across the arrays.


# ------------------------------------------------------------
# Basic indexing and slicing (1D)
# ------------------------------------------------------------

print("\n1D indexing and slicing:")
print("a[0] (first element):", a[0])
print("a[-1] (last element):", a[-1])
print("a[1:3] (index 1 and 2):", a[1:3])
print("a[:2] (first two):", a[:2])
print("a[2:] (from index 2 to end):", a[2:])

# Common mistake:
# In slicing [start:end], the end index is not included.


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

x = np.array([120, 35, 78, 210, 55])
print("\nBasic statistics:")
print("Mean:", x.mean())
print("Sum:", x.sum())
print("Max:", x.max())
print("Min:", x.min())


# ------------------------------------------------------------
# Matrices (2D arrays) and indexing (rows/columns)
# ------------------------------------------------------------

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\nMatrix M:")
print(M)
print("Shape of M:", M.shape)

print("\n2D indexing examples:")
print("M[0, 0] (row 0, col 0):", M[0, 0])
print("M[1, 2] (row 1, col 2):", M[1, 2])

print("\nSelecting rows/columns:")
print("M[0, :] (first row):", M[0, :])
print("M[:, 1] (second column):", M[:, 1])

print("\nSlicing a sub-matrix:")
print("M[0:2, 1:3] (rows 0-1, cols 1-2):")
print(M[0:2, 1:3])

# Explanation:
# In 2D indexing, the first index is the row and the second index is the column.


# ============================================================
# Pandas: DataFrames for business-style data analysis
# ============================================================

import pandas as pd


# ------------------------------------------------------------
# Create a DataFrame
# ------------------------------------------------------------
# Imagine this is an order-level dataset exported from an e-commerce system.
data = {
    "order_id":   [1001, 1002, 1003, 1004, 1005, 1006],
    "region":     ["North", "London", "North", "Midlands", "London", "Midlands"],
    "product":    ["Box", "Tape", "Mailer", "Box", "Label", "Tape"],
    "units":      [120, 30, 80, 200, 400, 20],
    "unit_price": [0.40, 1.20, 0.25, 0.40, 0.05, 1.20],
    "returns":    [0, 1, 0, 0, 0, 1]
}

df = pd.DataFrame(data)
print("\n--- DataFrame created ---")
print(df)


# ------------------------------------------------------------
# Quick inspection
# ------------------------------------------------------------
print("\n--- df.shape (rows, cols) ---")
print(df.shape)

print("\n--- df.head() ---")
print(df.head())

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() (numeric columns) ---")
print(df.describe())


# ------------------------------------------------------------
# Column selection
# ------------------------------------------------------------
print("\n--- Single column (Series): df['region'] ---")
print(df["region"])

print("\n--- Multiple columns (DataFrame): df[['order_id','region','units']] ---")
print(df[["order_id", "region", "units"]])


# ------------------------------------------------------------
# Create a new column (revenue)
# ------------------------------------------------------------
df["revenue"] = df["units"] * df["unit_price"]
print("\n--- Added revenue column ---")
print(df)


# ------------------------------------------------------------
# Row selection: iloc (by position) and loc (by label)
# ------------------------------------------------------------
print("\n--- iloc: first 3 rows, selected columns ---")
print(df.iloc[0:3, :])  # rows 0,1,2 and all columns

print("\n--- loc: filter rows by condition, then pick columns ---")
north_rows = df["region"] == "North"
print(df.loc[north_rows, ["order_id", "region", "product", "revenue"]])


# ------------------------------------------------------------
# Boolean filtering (selection rules)
# ------------------------------------------------------------
# Example: London orders with units >= 100
mask = (df["region"] == "London") & (df["units"] >= 100)
print("\n--- London orders with units >= 100 ---")
print(df.loc[mask])

# Example: returned orders (returns == 1)
print("\n--- Returned orders (returns == 1) ---")
print(df.loc[df["returns"] == 1])


# ------------------------------------------------------------
# Sorting (top revenue orders)
# ------------------------------------------------------------
print("\n--- Top 3 orders by revenue ---")
print(df.sort_values(by="revenue", ascending=False).head(3))


# ------------------------------------------------------------
# Groupby summary (business dashboard style)
# ------------------------------------------------------------
# Revenue by region
rev_by_region = df.groupby("region")["revenue"].sum()
print("\n--- Revenue by region (sum) ---")
print(rev_by_region)

# Average revenue per order by region
avg_rev_by_region = df.groupby("region")["revenue"].mean()
print("\n--- Average revenue per order by region ---")
print(avg_rev_by_region)

# Units by product
units_by_product = df.groupby("product")["units"].sum()
print("\n--- Units sold by product ---")
print(units_by_product)




# ============================================================
# Practical questions
# ============================================================

# Once you are familiar with the basics, these questions help you practise typical tasks.

# ------------------------------------------------------------
# Q1: variables + types
# ------------------------------------------------------------

# Task:
# Create three variables: your_name (string), age (int), is_attending_lecture (boolean), then print them.


# ------------------------------------------------------------
# Q2: arithmetic + rounding
# ------------------------------------------------------------

# Task:
# Given revenue = 82002 and orders = 205, compute AOV and print it with 2 decimal places.


# ------------------------------------------------------------
# Q3: if/elif/else decision rule
# ------------------------------------------------------------

# Task:
# Classify AOV:
# - AOV >= 400 -> "High AOV"
# - AOV >= 200 -> "Medium AOV"
# - else -> "Low AOV"


# ------------------------------------------------------------
# Q4: list append + loop accumulation
# ------------------------------------------------------------

# Task:
# Add "Bubble wrap" to products and 0.30 to costs, then recompute total_cost using a loop.
# Hint: use .append(value) to add item to a list


# ------------------------------------------------------------
# Q5: function + loop inside function
# ------------------------------------------------------------

# Task:
# Write mean_val(values) using a loop, then test it on order_values = [120, 35, 78, 210, 55].


# ------------------------------------------------------------
# Q6: NumPy array + shape + mean
# ------------------------------------------------------------

# Task:
# Create a NumPy array [2, 4, 6, 8, 10], then print its shape and mean.


# ------------------------------------------------------------
# Q7: maximum value (built-in vs loop)
# ------------------------------------------------------------

# Task:
# Find the maximum in nums = [3, 17, 9, 22, 5]
# (a) using max() function
# (b) using a loop without max()


# ------------------------------------------------------------
# Q8: function for circle area
# ------------------------------------------------------------

# Task:
# Write circle_area(radius) and test with radius = 3 and radius = 10.
# Hint: use np.pi and ** for power.


# ------------------------------------------------------------
# Q9: vectorised revenue calculation with NumPy
# ------------------------------------------------------------

# Task:
# Given units = [5, 10, 2] and price = [12, 7, 30],
# compute revenue per product and total revenue using NumPy.


# ------------------------------------------------------------
# Q10: standard deviation with NumPy
# ------------------------------------------------------------
# Task:
# Compute the standard deviation of order_values using NumPy.
# Hint: use .std()



# ------------------------------------------------------------
# Q11:
# ------------------------------------------------------------

# Task:
# Using df:
#   (a) Create df_high that keeps only orders with revenue >= 50
#   (b) Print only columns: order_id, region, revenue
# ------------------------------------------------------------


# ------------------------------------------------------------
# Q12:
# ------------------------------------------------------------

# Task:
# Using df:
#   (a) Compute revenue_by_product (sum of revenue for each product)
#   (b) Find the product with the largest total revenue
#   (c) Print both results

