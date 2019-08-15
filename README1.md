
Getting Started with pandas
In [1]:
import pandas as pd
In [2]:
from pandas import Series, DataFrame
In [3]:
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)
Introduction to pandas Data Structures
Series
In [4]:
obj = pd.Series([4, 7, -5, 3])
obj
Out[4]:
0    4
1    7
2   -5
3    3
dtype: int64
In [5]:
obj.values
obj.index  # like range(4)
Out[5]:
RangeIndex(start=0, stop=4, step=1)
In [6]:
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index
Out[6]:
Index(['d', 'b', 'a', 'c'], dtype='object')
In [7]:
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]
Out[7]:
c    3
a   -5
d    6
dtype: int64
In [8]:
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
Out[8]:
d     403.428793
b    1096.633158
a       0.006738
c      20.085537
dtype: float64
In [9]:
'b' in obj2
'e' in obj2
Out[9]:
False
In [10]:
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3
Out[10]:
Ohio      35000
Texas     71000
Oregon    16000
Utah       5000
dtype: int64
In [11]:
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4
Out[11]:
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64
In [12]:
pd.isnull(obj4)
pd.notnull(obj4)
Out[12]:
California    False
Ohio           True
Oregon         True
Texas          True
dtype: bool
In [13]:
obj4.isnull()
Out[13]:
California     True
Ohio          False
Oregon        False
Texas         False
dtype: bool
In [14]:
obj3
obj4
obj3 + obj4
Out[14]:
California         NaN
Ohio           70000.0
Oregon         32000.0
Texas         142000.0
Utah               NaN
dtype: float64
In [15]:
obj4.name = 'population'
obj4.index.name = 'state'
obj4
Out[15]:
state
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
Name: population, dtype: float64
In [16]:
obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj
Out[16]:
Bob      4
Steve    7
Jeff    -5
Ryan     3
dtype: int64
DataFrame
In [17]:
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
In [18]:
frame
Out[18]:
state	year	pop
0	Ohio	2000	1.5
1	Ohio	2001	1.7
2	Ohio	2002	3.6
3	Nevada	2001	2.4
4	Nevada	2002	2.9
5	Nevada	2003	3.2
In [19]:
frame.head()
Out[19]:
state	year	pop
0	Ohio	2000	1.5
1	Ohio	2001	1.7
2	Ohio	2002	3.6
3	Nevada	2001	2.4
4	Nevada	2002	2.9
In [20]:
pd.DataFrame(data, columns=['year', 'state', 'pop'])
Out[20]:
year	state	pop
0	2000	Ohio	1.5
1	2001	Ohio	1.7
2	2002	Ohio	3.6
3	2001	Nevada	2.4
4	2002	Nevada	2.9
5	2003	Nevada	3.2
In [21]:
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
frame2
frame2.columns
Out[21]:
Index(['year', 'state', 'pop', 'debt'], dtype='object')
In [22]:
frame2['state']
frame2.year
Out[22]:
one      2000
two      2001
three    2002
four     2001
five     2002
six      2003
Name: year, dtype: int64
In [23]:
frame2.loc['three']
Out[23]:
year     2002
state    Ohio
pop       3.6
debt      NaN
Name: three, dtype: object
In [24]:
frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(6.)
frame2
Out[24]:
year	state	pop	debt
one	2000	Ohio	1.5	0.0
two	2001	Ohio	1.7	1.0
three	2002	Ohio	3.6	2.0
four	2001	Nevada	2.4	3.0
five	2002	Nevada	2.9	4.0
six	2003	Nevada	3.2	5.0
In [25]:
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
Out[25]:
year	state	pop	debt
one	2000	Ohio	1.5	NaN
two	2001	Ohio	1.7	-1.2
three	2002	Ohio	3.6	NaN
four	2001	Nevada	2.4	-1.5
five	2002	Nevada	2.9	-1.7
six	2003	Nevada	3.2	NaN
In [26]:
frame2['eastern'] = frame2.state == 'Ohio'
frame2
Out[26]:
year	state	pop	debt	eastern
one	2000	Ohio	1.5	NaN	True
two	2001	Ohio	1.7	-1.2	True
three	2002	Ohio	3.6	NaN	True
four	2001	Nevada	2.4	-1.5	False
five	2002	Nevada	2.9	-1.7	False
six	2003	Nevada	3.2	NaN	False
In [27]:
del frame2['eastern']
frame2.columns
Out[27]:
Index(['year', 'state', 'pop', 'debt'], dtype='object')
In [28]:
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
In [29]:
frame3 = pd.DataFrame(pop)
frame3
Out[29]:
Nevada	Ohio
2000	NaN	1.5
2001	2.4	1.7
2002	2.9	3.6
In [30]:
frame3.T
Out[30]:
2000	2001	2002
Nevada	NaN	2.4	2.9
Ohio	1.5	1.7	3.6
In [33]:
# pd.DataFrame(pop, index=[2001, 2002, 2003])
In [34]:
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)
Out[34]:
Ohio	Nevada
2000	1.5	NaN
2001	1.7	2.4
In [35]:
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
Out[35]:
state	Nevada	Ohio
year		
2000	NaN	1.5
2001	2.4	1.7
2002	2.9	3.6
In [36]:
frame3.values
Out[36]:
array([[nan, 1.5],
       [2.4, 1.7],
       [2.9, 3.6]])
In [37]:
frame2.values
Out[37]:
array([[2000, 'Ohio', 1.5, nan],
       [2001, 'Ohio', 1.7, -1.2],
       [2002, 'Ohio', 3.6, nan],
       [2001, 'Nevada', 2.4, -1.5],
       [2002, 'Nevada', 2.9, -1.7],
       [2003, 'Nevada', 3.2, nan]], dtype=object)
Index Objects
In [38]:
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]
Out[38]:
Index(['b', 'c'], dtype='object')
index[1] = 'd' # TypeError

In [39]:
labels = pd.Index(np.arange(3))
labels
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels
Out[39]:
True
In [40]:
frame3
frame3.columns
'Ohio' in frame3.columns
2003 in frame3.index
Out[40]:
False
In [41]:
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels
Out[41]:
Index(['foo', 'foo', 'bar', 'bar'], dtype='object')
Essential Functionality
Reindexing
In [42]:
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
Out[42]:
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64
In [43]:
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
Out[43]:
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64
In [44]:
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
obj3.reindex(range(6), method='ffill')
Out[44]:
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
In [45]:
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=['a', 'c', 'd'],
                     columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
Out[45]:
Ohio	Texas	California
a	0.0	1.0	2.0
b	NaN	NaN	NaN
c	3.0	4.0	5.0
d	6.0	7.0	8.0
In [46]:
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
Out[46]:
Texas	Utah	California
a	1	NaN	2
c	4	NaN	5
d	7	NaN	8
In [47]:
frame.loc[['a', 'b', 'c', 'd'], states]
C:\Users\jonxia\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: 
Passing list-likes to .loc or [] with any missing label will raise
KeyError in the future, you can use .reindex() as an alternative.

See the documentation here:
https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
  """Entry point for launching an IPython kernel.
Out[47]:
Texas	Utah	California
a	1.0	NaN	2.0
b	NaN	NaN	NaN
c	4.0	NaN	5.0
d	7.0	NaN	8.0
Dropping Entries from an Axis
In [48]:
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])
Out[48]:
a    0.0
b    1.0
e    4.0
dtype: float64
In [49]:
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
Out[49]:
one	two	three	four
Ohio	0	1	2	3
Colorado	4	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
In [50]:
data.drop(['Colorado', 'Ohio'])
Out[50]:
one	two	three	four
Utah	8	9	10	11
New York	12	13	14	15
In [51]:
data.drop('two', axis=1)
data.drop(['two', 'four'], axis='columns')
Out[51]:
one	three
Ohio	0	2
Colorado	4	6
Utah	8	10
New York	12	14
In [52]:
obj.drop('c', inplace=True)
obj
Out[52]:
a    0.0
b    1.0
d    3.0
e    4.0
dtype: float64
Indexing, Selection, and Filtering
In [53]:
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]
Out[53]:
a    0.0
b    1.0
dtype: float64
In [54]:
obj['b':'c']
Out[54]:
b    1.0
c    2.0
dtype: float64
In [55]:
obj['b':'c'] = 5
obj
Out[55]:
a    0.0
b    5.0
c    5.0
d    3.0
dtype: float64
In [56]:
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
Out[56]:
three	one
Ohio	2	0
Colorado	6	4
Utah	10	8
New York	14	12
In [57]:
data[:2]
data[data['three'] > 5]
Out[57]:
one	two	three	four
Colorado	4	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
In [58]:
data < 5
data[data < 5] = 0
data
Out[58]:
one	two	three	four
Ohio	0	0	0	0
Colorado	0	5	6	7
Utah	8	9	10	11
New York	12	13	14	15
Selection with loc and iloc
In [59]:
data.loc['Colorado', ['two', 'three']]
Out[59]:
two      5
three    6
Name: Colorado, dtype: int32
In [60]:
data.iloc[2, [3, 0, 1]]
data.iloc[2]
data.iloc[[1, 2], [3, 0, 1]]
Out[60]:
four	one	two
Colorado	7	0	5
Utah	11	8	9
In [61]:
data.loc[:'Utah', 'two']
data.iloc[:, :3][data.three > 5]
Out[61]:
one	two	three
Colorado	0	5	6
Utah	8	9	10
New York	12	13	14
Integer Indexes
ser = pd.Series(np.arange(3.)) ser ser[-1]

In [62]:
ser = pd.Series(np.arange(3.))
In [63]:
ser
Out[63]:
0    0.0
1    1.0
2    2.0
dtype: float64
In [64]:
ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]
Out[64]:
2.0
In [65]:
ser[:1]
ser.loc[:1]
ser.iloc[:1]
Out[65]:
0    0.0
dtype: float64
Arithmetic and Data Alignment
In [66]:
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
               index=['a', 'c', 'e', 'f', 'g'])
s1
s2
Out[66]:
a   -2.1
c    3.6
e   -1.5
f    4.0
g    3.1
dtype: float64
In [67]:
s1 + s2
Out[67]:
a    5.2
c    1.1
d    NaN
e    0.0
f    NaN
g    NaN
dtype: float64
In [68]:
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
Out[68]:
b	d	e
Utah	0.0	1.0	2.0
Ohio	3.0	4.0	5.0
Texas	6.0	7.0	8.0
Oregon	9.0	10.0	11.0
In [69]:
df1 + df2
Out[69]:
b	c	d	e
Colorado	NaN	NaN	NaN	NaN
Ohio	3.0	NaN	6.0	NaN
Oregon	NaN	NaN	NaN	NaN
Texas	9.0	NaN	12.0	NaN
Utah	NaN	NaN	NaN	NaN
In [70]:
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1
df2
df1 - df2
Out[70]:
A	B
0	NaN	NaN
1	NaN	NaN
Arithmetic methods with fill values
In [71]:
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                   columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                   columns=list('abcde'))
df2.loc[1, 'b'] = np.nan
df1
df2
Out[71]:
a	b	c	d	e
0	0.0	1.0	2.0	3.0	4.0
1	5.0	NaN	7.0	8.0	9.0
2	10.0	11.0	12.0	13.0	14.0
3	15.0	16.0	17.0	18.0	19.0
In [72]:
df1 + df2
Out[72]:
a	b	c	d	e
0	0.0	2.0	4.0	6.0	NaN
1	9.0	NaN	13.0	15.0	NaN
2	18.0	20.0	22.0	24.0	NaN
3	NaN	NaN	NaN	NaN	NaN
In [73]:
df1.add(df2, fill_value=0)
Out[73]:
a	b	c	d	e
0	0.0	2.0	4.0	6.0	4.0
1	9.0	5.0	13.0	15.0	9.0
2	18.0	20.0	22.0	24.0	14.0
3	15.0	16.0	17.0	18.0	19.0
In [74]:
1 / df1
df1.rdiv(1)
Out[74]:
a	b	c	d
0	inf	1.000000	0.500000	0.333333
1	0.250000	0.200000	0.166667	0.142857
2	0.125000	0.111111	0.100000	0.090909
In [75]:
df1.reindex(columns=df2.columns, fill_value=0)
Out[75]:
a	b	c	d	e
0	0.0	1.0	2.0	3.0	0
1	4.0	5.0	6.0	7.0	0
2	8.0	9.0	10.0	11.0	0
Operations between DataFrame and Series
In [76]:
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]
Out[76]:
array([[0., 0., 0., 0.],
       [4., 4., 4., 4.],
       [8., 8., 8., 8.]])
In [77]:
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                     columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame
series
Out[77]:
b    0.0
d    1.0
e    2.0
Name: Utah, dtype: float64
In [78]:
frame - series
Out[78]:
b	d	e
Utah	0.0	0.0	0.0
Ohio	3.0	3.0	3.0
Texas	6.0	6.0	6.0
Oregon	9.0	9.0	9.0
In [79]:
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame + series2
Out[79]:
b	d	e	f
Utah	0.0	NaN	3.0	NaN
Ohio	3.0	NaN	6.0	NaN
Texas	6.0	NaN	9.0	NaN
Oregon	9.0	NaN	12.0	NaN
In [80]:
series3 = frame['d']
frame
series3
frame.sub(series3, axis='index')
Out[80]:
b	d	e
Utah	-1.0	0.0	1.0
Ohio	-1.0	0.0	1.0
Texas	-1.0	0.0	1.0
Oregon	-1.0	0.0	1.0
Function Application and Mapping
In [81]:
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)
Out[81]:
b	d	e
Utah	0.204708	0.478943	0.519439
Ohio	0.555730	1.965781	1.393406
Texas	0.092908	0.281746	0.769023
Oregon	1.246435	1.007189	1.296221
In [82]:
f = lambda x: x.max() - x.min()
frame.apply(f)
Out[82]:
b    1.802165
d    1.684034
e    2.689627
dtype: float64
In [83]:
frame.apply(f, axis='columns')
Out[83]:
Utah      0.998382
Ohio      2.521511
Texas     0.676115
Oregon    2.542656
dtype: float64
In [84]:
def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)
Out[84]:
b	d	e
min	-0.555730	0.281746	-1.296221
max	1.246435	1.965781	1.393406
In [85]:
format = lambda x: '%.2f' % x
frame.applymap(format)
Out[85]:
b	d	e
Utah	-0.20	0.48	-0.52
Ohio	-0.56	1.97	1.39
Texas	0.09	0.28	0.77
Oregon	1.25	1.01	-1.30
In [86]:
frame['e'].map(format)
Out[86]:
Utah      -0.52
Ohio       1.39
Texas      0.77
Oregon    -1.30
Name: e, dtype: object
Sorting and Ranking
In [87]:
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
Out[87]:
a    1
b    2
c    3
d    0
dtype: int64
In [88]:
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=1)
Out[88]:
a	b	c	d
three	1	2	3	0
one	5	6	7	4
In [89]:
frame.sort_index(axis=1, ascending=False)
Out[89]:
d	c	b	a
three	0	3	2	1
one	4	7	6	5
In [90]:
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
Out[90]:
2   -3
3    2
0    4
1    7
dtype: int64
In [91]:
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
Out[91]:
4   -3.0
5    2.0
0    4.0
2    7.0
1    NaN
3    NaN
dtype: float64
In [92]:
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_values(by='b')
Out[92]:
b	a
2	-3	0
3	2	1
0	4	0
1	7	1
In [93]:
frame.sort_values(by=['a', 'b'])
Out[93]:
b	a
2	-3	0
0	4	0
3	2	1
1	7	1
In [94]:
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
Out[94]:
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5
dtype: float64
In [95]:
obj.rank(method='first')
Out[95]:
0    6.0
1    1.0
2    7.0
3    4.0
4    3.0
5    2.0
6    5.0
dtype: float64
In [96]:
# Assign tie values the maximum rank in the group
obj.rank(ascending=False, method='max')
Out[96]:
0    2.0
1    7.0
2    2.0
3    4.0
4    5.0
5    6.0
6    4.0
dtype: float64
In [97]:
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                      'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis='columns')
Out[97]:
b	a	c
0	3.0	2.0	1.0
1	3.0	1.0	2.0
2	1.0	2.0	3.0
3	3.0	2.0	1.0
Axis Indexes with Duplicate Labels
In [98]:
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
Out[98]:
a    0
a    1
b    2
b    3
c    4
dtype: int64
In [99]:
obj.index.is_unique
Out[99]:
False
In [100]:
obj['a']
obj['c']
Out[100]:
4
In [101]:
df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df.loc['b']
Out[101]:
0	1	2
b	1.669025	-0.438570	-0.539741
b	0.476985	3.248944	-1.021228
Summarizing and Computing Descriptive Statistics
In [102]:
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df
Out[102]:
one	two
a	1.40	NaN
b	7.10	-4.5
c	NaN	NaN
d	0.75	-1.3
In [103]:
df.sum()
Out[103]:
one    9.25
two   -5.80
dtype: float64
In [104]:
df.sum(axis='columns')
Out[104]:
a    1.40
b    2.60
c    0.00
d   -0.55
dtype: float64
In [105]:
df.mean(axis='columns', skipna=False)
Out[105]:
a      NaN
b    1.300
c      NaN
d   -0.275
dtype: float64
In [106]:
df.idxmax()
Out[106]:
one    b
two    d
dtype: object
In [107]:
df.cumsum()
Out[107]:
one	two
a	1.40	NaN
b	8.50	-4.5
c	NaN	NaN
d	9.25	-5.8
In [108]:
df.describe()
Out[108]:
one	two
count	3.000000	2.000000
mean	3.083333	-2.900000
std	3.493685	2.262742
min	0.750000	-4.500000
25%	1.075000	-3.700000
50%	1.400000	-2.900000
75%	4.250000	-2.100000
max	7.100000	-1.300000
In [109]:
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
Out[109]:
count     16
unique     3
top        a
freq       8
dtype: object
