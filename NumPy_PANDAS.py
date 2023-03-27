"Working with NumPy and PANDAS"
"https://www.youtube.com/watch?v=r-uOLxNrNk8"

import sys
import time

import numpy as np
import pandas as pd

"NumPy"
#    NumPy is an array processing library
#        Adds fancy math operations so you dont have to
#    NumPy allows for 'low level' instruction processing
#        *In A Nutshell*   makes run time lower than normal Python code
     

#    Effeciancy demonstration....
"""
print("Job ===> Square elements (0 -> 99,999) and sum them. Print the results \n\n\n")
start = time.time()
l = list(range(100000))
print("Sum = ", sum([x ** 2 for x in l]))
print("Normal Python => ", time.time() - start)
print("\n")
start2 = time.time()
a = np.arange(100000)
print("Sum = ", np.sum(a ** 2))
print("NumPy Python => ", time.time() - start2)
"""


"NumPy.int~~()"
#    np.int has a cap of 64
#    highest possible is np.int64()
#    increment in powers of 2. 2, 4, 8, 16, 32, 64
#    9,223,372,036,854,775,807 => largest assignable int (with 64 bits)
"""
print("NumPy.int~~()\n\n")
i = 0
while i <= 124:
    x = np.int8(i)
    print(x)
    i += 4
"""
###############################################################################


"NumPy.array([~,~,~])"
#    arrays work similar to python arrays
#    NumPy arrays have types.... types can be forced and modified
#    avoid storing characters and strings in a NumPy array
"""
print("NumPy.array([~,~,~])\n\n")
a = np.array([1, 2, 3, 4, 5])
print("a = ", a)
b = np.array([1, 2.3, 3.5, 5.8, 8.13])
print("b = ", b)
print("a[2:] = ", a[2:])
print("b[1:-1] = ", b[1:-1])
print("a[[0,2,4]] = ", a[[0,2,4]], "       --> this is a NumPy feature")
print("a.dtype = ", a.dtype)
print("b.dtype = ", b.dtype, "\n")
c = np.array([6, 7, 8, 9, 0], dtype=np.int8)
print("c = ", c)
print("c.dtype = ", c.dtype, "       --> this is a NumPy feature")
d = np.array([6, 7, 8, 9, 0], dtype=np.float16)
print("d = ", d)
print("d.dtype = ", d.dtype, "       --> this is a NumPy feature")
"""
###############################################################################


"NumPy.array([[~,~,~], [~,~,~], [~,~,~]])"
#    arrays work similar to python arrays
#    NumPy allows for matrices to be created and used
#        Matrix[d1, d2, d3, d4] allows you to choose a singular value from a matrix as if it were written as Matrix[d1][d2][d3][d4]
#    keep the matrix dimensions consistant, or the matrix will default to an object
#    NumPy auto expands values as demonstrated below
"""
print("NumPy.array([[~,~,~], [~,~,~], [~,~,~]])\n\n")
A = np.array(
    [[1, 2, 3],      #0
    [4, 5, 6],       #1
    [7, 8, 9]])     #2
#  0  1  2

print("A = ", A, "\n")
print("A.shape = ", A.shape, "      states the shape of the matrix. AKA ..... 2x3 matrix, 7x7 matrix, 1x20347028 matrix")
print("A.ndim = ", A.ndim,
"      states the dimensions of the matrix. AKA ..... [[~,~], [~,~], [~,~]] is 2, [[[~,~],[~,~]],[[~,~],[~,~]], [[~,~],[~,~]]] is 3")
print("A.size = ", A.size, "      states the number of elements in the matrix")
print("A[0] = ", A[0])
print("A[0][0] = ", A[0][0])
print("A[2, 2] = ", A[2, 2], "\n")

A[1] = [0, 0, 0]
print("A[1] = [0, 0, 0] -> \n", A, "\n")
A[2] = 24
print("A[2] = 24 -> \n", A)
"""
###############################################################################


"NumPy statistics"
#    has common statistics methods built in
#    axis refers to the dimensions of the array, starting at 0 for 1 dim, 1 for 2 dim and so on....
#        for 2D ==> 0 -> rows   1 -> columns   2 -> out of bounds
"""
print("NumPy statistics\n\n")
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("a = ", a)
print("a.sum() = ", a.sum())
print("a.mean() = ", a.mean())
print("a.std() = ", a.std())
print("a.var() = ", a.var())

A = np.array(
    [[1, 2, 3],      #0
    [4, 5, 6],       #1
    [7, 8, 9]])     #2
#  0  1  2
print("A = ", A)
print("A.sum(axis=0) = ", A.sum(axis=0), "      -> columns")
print("A.sum(axis=1) = ", A.sum(axis=1), "      -> rows")
print("A.mean(axis=2) = ", A.mean(axis=0), "      -> columns")
print("A.std(axis=0) = ", A.std(axis=0), "      -> columns")
print("A.var(axis=1) = ", A.var(axis=1), "      -> rows")
"""
###############################################################################


"NumPy.arange(~)"
#    creates a sorted array of range ~, non inclusive
#    using vectorizing operations does not change these arrays, rather creates and returns new arrays post vectorization operations
#        '+', '-', '*' are vectorization operations. '+=', '-=' are not vectorization operations and will modify the original array
"""
print("NumPy.arange(~)\n\n")
a = np.arange(4)
b = np.arange(7)
c = np.array([12, 12, 12, 12])
print("a = np.arange(4) ==> ", a)
print("b = np.arange(7) ==> ", b)
print("c = np.array([12, 12, 12, 12]) ==> ", c)
print("a + 10 = ", a + 10, "      -> This is a vectorizing operation. AKA this operation is applied to all the elements in the array")
print("a * 10 = ", a * 10, "      -> This is a vectorizing operation. AKA this operation is applied to all the elements in the array")
print("a / 10 = ", a / 10, "      -> \" This is a vectorizing ...                                                                                                                          array \"")
print("a + c = ", a + c)
print("a + b = \'syntax error, different sizes  --> a(4), b(7)\'")
print("b + b = ", b + b)
print("a * c = ", a * c)
"""
###############################################################################


"Boolean Arrays"
#    similar to vectorized arrays
"""
print("Boolean arrays                          *notice parallels*\n\n")
a = np.arange(4)
t = True
f = False
print("t = True, f = False")
print("a = np.arange(4) ==> ", a)
print("a[[0, -1]] = ", a[[0, -1]], "        which is equal to        ", "a[[t, f, f, t]] = ", a[[t, f, f, t]],
"                  ---> this is a way of selecting data")
print("a >= 2    ==>    ", a >= 2)
print("a[[a >= 2]] = ", a[[a >= 2]])
print("***all operations that result in a boolean value can be used to create the conditions of the array***")
"""
###############################################################################


"Linear Algebra"
#    all linear algebra operations are built into NumPy
"""
print("Linear Algebra\n\n")
A = np.array(
    [[1, 2, 3],      #0
    [4, 5, 6],       #1
    [7, 8, 9]])     #2
#  0  1  2
B = np.array(
    [[6, 5],      #0
    [4, 3],       #1
    [2, 1]])     #2
#  0  1

print("A = ", A, "\n")
print("B = ", B, "\n\n")
print("A.dot(B) = ", A.dot(B))      # dot product
print("A @ B = ", A @ B)             # cross product
print("B.T = ", B.T)                      # transposing '~' matrix
"""
###############################################################################

###############################################################################

###############################################################################

###############################################################################



"PANDAS"
#    PANDAS is a data analysis and processing library
#    For the most part, operations are immutable


"PANDAS.Series(~)"
#    indices in PANDAS are normally assigned as 0->n but can be modified to other values, such as names.
#        values in PANDAS series can be refrenced by their index
#    all NumPy operations can be used on PANDAS series
"""
print("PANDAS.Series(~)\n\n")
g7_pop = pd.Series([35.467, 63.951, 80.940, 60.665, 127.061, 64.511, 318.523])
g7_pop.name = "G7 Populations (in millions)"
print("g7_pop.name = \"G7 Populations (in millions)\" ==> \n", g7_pop)
print("\n")
print("g7_pop.dtype = ", g7_pop.dtype)
print("g7_pop.values = ", g7_pop.values)
print("g7_pop.index = ", g7_pop.index)
g7_pop.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]
print("\ng7_pop (after index modification, see code) = \n", g7_pop)
print("\n")
print("g7_pop['Canada'] = ", g7_pop['Canada'])
print("g7_pop[0] = ", g7_pop[0])
print("g7_pop['Germany'] = ", g7_pop['Germany'])
print("\n")
print("g7_pop[['Japan', 'United States', 'Italy']] = \n", g7_pop[['Japan', 'United States', 'Italy']])
"""
###############################################################################


"DataFrames"
#    spreadsheet display format
#    combination of multiple series
"""
print("DataFrames\n\n")
df = pd.DataFrame({
    'Population': [35.467, 63.951, 80.94 , 60.665, 127.061, 64.511, 318.523],
    'GDP': [
        1785387,
        2833687,
        3874437,
        2167744,
        4602367,
        2950039,
        17348075
    ],
    'Surface Area': [
        9984670,
        640679,
        357114,
        301336,
        377930,
        242495,
        9525067
    ],
    'HDI': [
        0.913,
        0.888,
        0.916,
        0.873,
        0.891,
        0.907,
        0.915
    ],
    'Continent': [
        'America',
        'Europe',
        'Europe',
        'Europe',
        'Asia',
        'Europe',
        'America'
    ]
}, columns=['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])
print(df, "\n")
print("df.info() = ")
print(df.info(), "\n")
print("df.size = ", df.size)
print("df.shape = ", df.shape)
print("\n df.describe() => \n", df.describe(), "\n")       # gives statistical summary of the item being described

df.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]
print("df.loc['Canada'] = \n", df.loc['Canada'])      # select row by index
print()
print("df.iloc[-1] = \n", df.iloc[-1])      # select row by sequential position
print()
print("df['Population'] = \n", df['Population'])      # select column by name
print()
print("df.loc['France': 'Italy', ['Population', 'GDP']] = \n", df.loc['France': 'Italy', ['Population', 'GDP']])
"""
###############################################################################


"Dropping Items"

"""
print("Dropping Items\n\n")
df = pd.DataFrame({
    'Population': [35.467, 63.951, 80.94 , 60.665, 127.061, 64.511, 318.523],
    'GDP': [
        1785387,
        2833687,
        3874437,
        2167744,
        4602367,
        2950039,
        17348075
    ],
    'Surface Area': [
        9984670,
        640679,
        357114,
        301336,
        377930,
        242495,
        9525067
    ],
    'HDI': [
        0.913,
        0.888,
        0.916,
        0.873,
        0.891,
        0.907,
        0.915
    ],
    'Continent': [
        'America',
        'Europe',
        'Europe',
        'Europe',
        'Asia',
        'Europe',
        'America'
    ]
}, columns=['Population', 'GDP', 'Surface Area', 'HDI', 'Continent'])
df.index = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]

print("df =\n", df)
print()
print("df.drop(['Canada', 'Japan', 'Italy']) =\n", df.drop(['Canada', 'Japan', 'Italy']))
print()
print("df.drop(columns=['Population', 'GDP', 'HDI']) =\n", df.drop(columns=['Population', 'GDP', 'HDI']))
print()
print("crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])")
crisis = pd.Series([-1_000_000, -0.3], index=['GDP', 'HDI'])
print("df[['GDP', 'HDI']] + crisis = \n", df[['GDP', 'HDI']] + crisis)
langs = pd.Series(
    ['French', 'German', 'Italian'],
    index=['France', 'Germany', 'Italy'],
    name='Language'
)
df['Language'] = langs
print("df after adding language column = \n", df)      # NaN === not assigned
"""
###############################################################################


"Input/Output tools"
#    Refer to https://pandas.pydata.org/docs/pandas.pdf for documentation on said tools

###############################################################################