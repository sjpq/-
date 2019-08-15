# -In [10]:
!cat examples/ex1.csv

!cat examples/ex2.csv
a,b,c,d,message
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
In [17]:
import numpy as np
import pandas as pd
df = pd.read_csv('examples/ex1.csv')
print(df)

#无header，添加column headers，添加index_col
print(pd.read_csv('examples/ex2.csv',header=None))
print(pd.read_csv('examples/ex2.csv',names = ['a', 'b', 'c', 'd', 'message'], index_col = 'message'))
   a   b   c   d message
0  1   2   3   4   hello
1  5   6   7   8   world
2  9  10  11  12     foo
   0   1   2   3      4
0  1   2   3   4  hello
1  5   6   7   8  world
2  9  10  11  12    foo
         a   b   c   d
message               
hello    1   2   3   4
world    5   6   7   8
foo      9  10  11  12
In [32]:
!cat examples/csv_mindex.csv
print('######')
parsed = pd.read_csv('examples/csv_mindex.csv', index_col = ['key1','key2'])
print(parsed)
#如果1-多个空格为间隔，那么可以用sep = 's+'来分隔。
!cat examples/ex4.csv
parsed1 = pd.read_csv('examples/ex4.csv',skiprows = [0,2,3])
print('\n')
print(parsed1)
print('##################')
!cat examples/ex5.csv
print('\n')
#na_values可以把一串strings定义为missing value.
sentinels = {'message':['NA','foo'],'something':['two']}
parsed2 = pd.read_csv('examples/ex5.csv',na_values=sentinels)
print(parsed2)
key1,key2,value1,value2
one,a,1,2
one,b,3,4
one,c,5,6
one,d,7,8
two,a,9,10
two,b,11,12
two,c,13,14
two,d,15,16
######
           value1  value2
key1 key2                
one  a          1       2
     b          3       4
     c          5       6
     d          7       8
two  a          9      10
     b         11      12
     c         13      14
     d         15      16
# hey!
a,b,c,d,message
# just wanted to make things more difficult for you 
# who reads CSV files with computers, anyway? 
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo

   a   b   c   d message
0  1   2   3   4   hello
1  5   6   7   8   world
2  9  10  11  12     foo
##################
something,a,b,c,d,message
one,1,2,3,4,NA
two,5,6,,8,world
three,9,10,11,12,foo

  something  a   b     c   d message
0       one  1   2   3.0   4     NaN
1       NaN  5   6   NaN   8   world
2     three  9  10  11.0  12     NaN
In [56]:
print('########################Reading Text Files in Pieces##########################')
pd.options.display.max_rows = 10  #对于大文件仅显示十列，或者可以nrows = 10来显示
chunker = pd.read_csv('../examples/ex6.csv',chunksize = 1000)
tot = pd.Series([]) #构建一个空series，然后每块piece的出现的次数相加
#对每小块文件对key列统计值出现的次数
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value = 0)
tot = tot.sort_values(ascending=False)
print(tot)

print('########## Wrting Data to Text Format##########################')
data = pd.read_csv('../examples/ex5.csv')
print(data, '\n')
data.to_csv('examples/out.csv')
#sys.stdout可以把内容输出到console，比如可以用sep = 来做间隔，对于missing value，可以用na_rep = 'NULL'等方式来替代
import sys
print(data.to_csv(sys.stdout, sep = '|'),'\n')
print(data.to_csv(sys.stdout, na_rep = 'NULL'),'\n')
#行名列名如果不指定都输出；设定index和header可以调整
print(data.to_csv(sys.stdout, index = False, header = False,sep='|'), '\n')
print(data.to_csv(sys.stdout, index = False, columns = ['a','b','c']), '\n')

dates = pd.date_range('1/1/2000',periods=7)
ts = pd.Series(np.arange(7), index = dates)
ts.to_csv('examples/tseries.csv')
!cat examples/tseries.csv
########################Reading Text Files in Pieces##########################
E    368.0
X    364.0
L    346.0
O    343.0
Q    340.0
     ...  
5    157.0
2    152.0
0    151.0
9    150.0
1    146.0
Length: 36, dtype: float64
########## Wrting Data to Text Format##########################
  something  a   b     c   d message
0       one  1   2   3.0   4     NaN
1       two  5   6   NaN   8   world
2     three  9  10  11.0  12     foo 

|something|a|b|c|d|message
0|one|1|2|3.0|4|
1|two|5|6||8|world
2|three|9|10|11.0|12|foo
None 

,something,a,b,c,d,message
0,one,1,2,3.0,4,NULL
1,two,5,6,NULL,8,world
2,three,9,10,11.0,12,foo
None 

one|1|2|3.0|4|
two|5|6||8|world
three|9|10|11.0|12|foo
None 

a,b,c
1,2,3.0
5,6,
9,10,11.0
None 

2000-01-01,0
2000-01-02,1
2000-01-03,2
2000-01-04,3
2000-01-05,4
2000-01-06,5
2000-01-07,6
/Users/theomarker/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:25: FutureWarning: The signature of `Series.to_csv` was aligned to that of `DataFrame.to_csv`, and argument 'header' will change its default value from False to True: please pass an explicit value to suppress this warning.
In [75]:
print('########## Wrting Data to Text Format##########################')
import csv
with open('../examples/ex7.csv') as f:
    lines = list(csv.reader(f))
header, values = lines[0], lines[1:]
data_dict = {h: v   for h,v in zip(header, values)}
print(data_dict)
print('########## JSON Data ##########################')
obj = """
    {"name": "Wes",
     "places_lived": ["United States", "Spain", "Germany"],
     "pet": null,
     "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
                  {"name": "Katie", "age": 38,
                   "pets": ["Sixes", "Stache", "Cisco"]}]
} """
import json
result = json.loads(obj)
print(result['siblings'])
#json.dumps可以把其从一个python object转变为JSON格式
asjson = json.dumps(result)
print(asjson)
siblings = pd.DataFrame(result['siblings'],columns = ['name','age'])
print(siblings)
!cat ../examples/example.json
data = pd.read_json('../examples/example.json')
print(data)
print(data.to_json())
print(data.to_json(orient='records'))
print('########## XML and HTML: Web Scraping ##########################')
#pandas.read_html在default下会搜寻tabular data contained within <table> tags.结果是dataframe object
tables = pd.read_html('../examples/fdic_failed_bank_list.html')
len(tables)
failures = tables[0]
print(failures.head())
close_timestamps = pd.to_datetime(failures['Closing Date'])
close_timestamps.dt.year.value_counts()
########## Wrting Data to Text Format##########################
{'a': ['1', '2', '3'], 'b': ['1', '2', '3']}
########## JSON Data ##########################
[{'name': 'Scott', 'age': 30, 'pets': ['Zeus', 'Zuko']}, {'name': 'Katie', 'age': 38, 'pets': ['Sixes', 'Stache', 'Cisco']}]
{"name": "Wes", "places_lived": ["United States", "Spain", "Germany"], "pet": null, "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]}, {"name": "Katie", "age": 38, "pets": ["Sixes", "Stache", "Cisco"]}]}
    name  age
0  Scott   30
1  Katie   38
[{"a": 1, "b": 2, "c": 3},
 {"a": 4, "b": 5, "c": 6},
 {"a": 7, "b": 8, "c": 9}]
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
{"a":{"0":1,"1":4,"2":7},"b":{"0":2,"1":5,"2":8},"c":{"0":3,"1":6,"2":9}}
[{"a":1,"b":2,"c":3},{"a":4,"b":5,"c":6},{"a":7,"b":8,"c":9}]
########## XML and HTML: Web Scraping ##########################
                      Bank Name             City  ST   CERT  \
0                   Allied Bank         Mulberry  AR     91   
1  The Woodbury Banking Company         Woodbury  GA  11297   
2        First CornerStone Bank  King of Prussia  PA  35312   
3            Trust Company Bank          Memphis  TN   9956   
4    North Milwaukee State Bank        Milwaukee  WI  20364   

                 Acquiring Institution        Closing Date       Updated Date  
0                         Today's Bank  September 23, 2016  November 17, 2016  
1                          United Bank     August 19, 2016  November 17, 2016  
2  First-Citizens Bank & Trust Company         May 6, 2016  September 6, 2016  
3           The Bank of Fayette County      April 29, 2016  September 6, 2016  
4  First-Citizens Bank & Trust Company      March 11, 2016      June 16, 2016  
Out[75]:
2010    157
2009    140
2011     92
2012     51
2008     25
       ... 
2004      4
2001      4
2007      3
2003      3
2000      2
Name: Closing Date, Length: 15, dtype: int64
2. Binary Data Formats
In [91]:
#to_pickle可以将数据存为pickle的二进制格式
frame = pd.read_csv('../examples/ex1.csv')
#存为2进制格式
frame.to_pickle('examples/frame_pickle')
pd.read_pickle('examples/frame_pickle')
print('########## Using HDF5 Format ##########################')
frame = pd.DataFrame({'a':np.random.randn(100)})
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store['obj1']
print('##########Reading Microsoft Excel Files##########################')
xlsx = pd.ExcelFile('../examples/ex1.xlsx')
print(pd.read_excel(xlsx, 'Sheet1'))
frame = pd.read_excel('../examples/ex1.xlsx', 'Sheet1')
frame
#将pandas数据导入成Excel格式，首先要创建一个ExcelWriter，然后用to_excel
writer = pd.ExcelWriter('../examples/ex1.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()
frame.to_excel('examples/ex2.xlsx')
########## Using HDF5 Format ##########################
##########Reading Microsoft Excel Files##########################
   Unnamed: 0  a   b   c   d message
0           0  1   2   3   4   hello
1           1  5   6   7   8   world
2           2  9  10  11  12     foo
3.interacting with web APIs
In [101]:
#链接API信息可以用requests package
import requests
url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
data = resp.json()
#print(data)
data[0]['title']
issues = pd.DataFrame(data, columns = ['number','title','labels','state'])
issues
Out[101]:
number	title	labels	state
0	27868	CI	[]	open
1	27867	CI: Failing to compile for Travis 3.7 build	[]	open
2	27866	Added missing space to error description	[]	open
3	27865	Cannot use .ix in IntervaIndex('pandas._libs.i...	[{'id': 2822098, 'node_id': 'MDU6TGFiZWwyODIyM...	open
4	27862	DOC: See also for DataFrame.iterrows() should ...	[{'id': 134699, 'node_id': 'MDU6TGFiZWwxMzQ2OT...	open
...	...	...	...	...
25	27827	BUG: Fixed groupby quantile for listlike q	[{'id': 233160, 'node_id': 'MDU6TGFiZWwyMzMxNj...	open
26	27826	BUG: Fix groupby quantile segfault	[{'id': 233160, 'node_id': 'MDU6TGFiZWwyMzMxNj...	open
27	27825	Dispatch pd.is_na for scalar extension value	[{'id': 849023693, 'node_id': 'MDU6TGFiZWw4NDk...	open
28	27824	Document DataFrame._constructor_sliced-like pr...	[{'id': 35818298, 'node_id': 'MDU6TGFiZWwzNTgx...	open
29	27820	DOC: clarify that read_parquet accepts a direc...	[{'id': 134699, 'node_id': 'MDU6TGFiZWwxMzQ2OT...	open
30 rows × 4 columns

interacting with databases
In [1]:
#pandas有一些简单的功能可以把从SQL里提取数据方便化
import pandas as pd
import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL,        d INTEGER
);"""
print(query)
con = sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL,        d INTEGER
);
---------------------------------------------------------------------------
OperationalError                          Traceback (most recent call last)
<ipython-input-1-07b6b21e8627> in <module>()
      9 print(query)
     10 con = sqlite3.connect('mydata.sqlite')
---> 11 con.execute(query)
     12 con.commit()

OperationalError: table test already exists
In [2]:
data = [('Atlanta', 'Georgia', 1.25, 6),
         ('Tallahassee', 'Florida', 2.6, 3),
         ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()
In [3]:
cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows
Out[3]:
[('Atlanta', 'Georgia', 1.25, 6),
 ('Tallahassee', 'Florida', 2.6, 3),
 ('Sacramento', 'California', 1.7, 5)]
In [4]:
cursor.description
pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
Out[4]:
a	b	c	d
0	Atlanta	Georgia	1.25	6
1	Tallahassee	Florida	2.60	3
2	Sacramento	California	1.70	5
In [5]:
import sqlalchemy as sqla
db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)
Out[5]:
a	b	c	d
0	Atlanta	Georgia	1.25	6
1	Tallahassee	Florida	2.60	3
2	Sacramento	California	1.70	5
