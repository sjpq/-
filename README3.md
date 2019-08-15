数据清洗和准备

处理缺失数据
滤除缺失数据
填充缺失数据
数据转换
利用函数或映射进行数据转换
替换值
重命名轴索引
离散化和面元划分
检测和过滤异常值
排列和随机采样
字符串操作
字符串对象方法
处理缺失数据

import numpy as np
string_data = pd.Series(['arrdvark','artichoke',np.nan,'avocado'])
string_data
>>>
0     arrdvark
1    artichoke
2          NaN
3      avocado
dtype: object
In [3]:

string_data.isnull()
Out[3]:
0    False
1    False
2     True
3    False
dtype: bool
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
缺失数据处理的函数：


滤除缺失数据

from numpy import nan as NA
data = pd.Series([1,NA,3.5,NA,7])
data.dropna()
>>>
0    1.0
2    3.5
4    7.0
dtype: float64
1
2
3
4
5
6
7
8
等价于：

data[data.notnull()]
>>>
0    1.0
2    3.5
4    7.0
dtype: float64
1
2
3
4
5
6
# dropna默认丢弃任何含有缺失值的行
data = pd.DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
data.dropna()
>>>

0	1	2
0	1.0	6.5	3.0
In [9]:

# 传入how = ‘all’只丢弃全为NA的行
data.dropna(how = 'all')
Out[9]:
0	1	2
0	1.0	6.5	3.0
1	1.0	NaN	NaN
3	NaN	6.5	3.0
In [10]:

# 丢弃全为NA的列
data[4] = NA
data.dropna(how = 'all',axis = 1)
Out[10]:
0	1	2
0	1.0	6.5	3.0
1	1.0	NaN	NaN
2	NaN	NaN	NaN
3	NaN	6.5	3.0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
# thresh参数指定删除行
df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:4,1] = NA
df.iloc[:2,2] = NA
df
>>>

0	1	2
0	-0.706974	NaN	NaN
1	0.132236	NaN	NaN
2	-0.023318	NaN	0.983925
3	-0.226226	NaN	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
In [12]:

df.dropna()
Out[12]:
0	1	2
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
In [13]:

df.dropna(thresh = 2)
Out[13]:
0	1	2
2	-0.023318	NaN	0.983925
3	-0.226226	NaN	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
填充缺失数据

df.fillna(0)
>>>

0	1	2
0	-0.706974	0.000000	0.000000
1	0.132236	0.000000	0.000000
2	-0.023318	0.000000	0.983925
3	-0.226226	0.000000	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
In [15]:

# 对不同的列填充不同的值
df.fillna({1:0.5,2:0})
Out[15]:
0	1	2
0	-0.706974	0.500000	0.000000
1	0.132236	0.500000	0.000000
2	-0.023318	0.500000	0.983925
3	-0.226226	0.500000	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
In [16]:

df
Out[16]:
0	1	2
0	-0.706974	NaN	NaN
1	0.132236	NaN	NaN
2	-0.023318	NaN	0.983925
3	-0.226226	NaN	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
In [18]:

# fillna默认返回新对象，但也可以对现有对象修改
_ = df.fillna(0,inplace = True)
df
Out[18]:
0	1	2
0	-0.706974	0.000000	0.000000
1	0.132236	0.000000	0.000000
2	-0.023318	0.000000	0.983925
3	-0.226226	0.000000	0.171830
4	-0.738432	-1.719353	-1.061145
5	-1.376627	0.327799	1.637936
6	-1.329905	-0.184855	0.400009
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
df = pd.DataFrame(np.random.randn(6,3))
df.iloc[2:,1] = NA
df.iloc[4:,2] = NA
df
>>>

0	1	2
0	-1.283371	0.620279	2.901213
1	1.180507	-0.460727	1.506817
2	0.690744	NaN	0.044961
3	-1.679425	NaN	1.070538
4	0.060521	NaN	NaN
5	1.778530	NaN	NaN
In [21]:

df.fillna(method ='ffill')
Out[21]:
0	1	2
0	-1.283371	0.620279	2.901213
1	1.180507	-0.460727	1.506817
2	0.690744	-0.460727	0.044961
3	-1.679425	-0.460727	1.070538
4	0.060521	-0.460727	1.070538
5	1.778530	-0.460727	1.070538
In [22]:

df.fillna(method = 'ffill',limit = 2)
Out[22]:
0	1	2
0	-1.283371	0.620279	2.901213
1	1.180507	-0.460727	1.506817
2	0.690744	-0.460727	0.044961
3	-1.679425	-0.460727	1.070538
4	0.060521	NaN	1.070538
5	1.778530	NaN	1.070538
In [23]:

# 传入Series的平均值
data = pd.Series([1.,NA,3.5,NA,7])
data.fillna(data.mean())
Out[23]:
0    1.000000
1    3.833333
2    3.500000
3    3.833333
4    7.000000
dtype: float64
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
fillna函数的参数：


数据转换

# duplicated方法返回一个布尔型Series，表示各行是否有重复行
data = pd.DataFrame({'k1':['one','two']*3+['two'],'k2':[1,1,2,3,3,4,4]})
data
>>>

k1	k2
0	one	1
1	two	1
2	one	2
3	two	3
4	one	3
5	two	4
6	two	4
In [27]:

data.duplicated() # 重复值标记为True
Out[27]:
0    False
1    False
2    False
3    False
4    False
5    False
6     True
dtype: bool
In [30]:

# driop_duplicats方法直接删除重复行，返回一个DataFrame
data.drop_duplicates()
Out[30]:
k1	k2
0	one	1
1	two	1
2	one	2
3	two	3
4	one	3
5	two	4
In [31]:

# 指定部分列进行重复项判断
data['v1'] = range(7)
data.drop_duplicates(['k1'])  # drop_duplicates和duplicated默认保留第一个出现的值
Out[31]:
k1	k2	v1
0	one	1	0
1	two	1	1
In [32]:

data.drop_duplicates(['k1','k2'],keep = 'last')
Out[32]:
k1	k2	v1
0	one	1	0
1	two	1	1
2	one	2	2
3	two	3	3
4	one	3	4
6	two	4	6
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
利用函数或映射进行数据转换

data = pd.DataFrame({'food': ['bacon', 'pulled pork','bacon','Pastrami', 'corned beef','Bacon', 'pastrami', 'honey ham', 'novalox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5,6]})
data
>>>

food	ounces
0	bacon	4.0
1	pulled pork	3.0
2	bacon	12.0
3	Pastrami	6.0
4	corned beef	7.5
5	Bacon	8.0
6	pastrami	3.0
7	honey ham	5.0
8	novalox	6.0
In [37]:

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
# food首字母统一小写
lowercased = data['food'].str.lower()
lowercased
Out[37]:
0          bacon
1    pulled pork
2          bacon
3       pastrami
4    corned beef
5          bacon
6       pastrami
7      honey ham
8        novalox
Name: food, dtype: object
In [38]:

data['animal'] = lowercased.map(meat_to_animal)
data
Out[38]:
food	ounces	animal
0	bacon	4.0	pig
1	pulled pork	3.0	pig
2	bacon	12.0	pig
3	Pastrami	6.0	cow
4	corned beef	7.5	cow
5	Bacon	8.0	pig
6	pastrami	3.0	cow
7	honey ham	5.0	pig
8	novalox	6.0	NaN
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
替换值

data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
>>>
0       1.0
1    -999.0
2       2.0
3    -999.0
4   -1000.0
5       3.0
dtype: float64
In [40]:

data.replace(-999,np.nan)
Out[40]:
0       1.0
1       NaN
2       2.0
3       NaN
4   -1000.0
5       3.0
dtype: float64
In [42]:

# 替换多个值
data.replace([-999,-1000],np.nan)
Out[42]:
0    1.0
1    NaN
2    2.0
3    NaN
4    NaN
5    3.0
dtype: float64
In [43]:

# 每个值对应不同的替换值
data.replace([-999,-1000],[np.nan,0])
Out[43]:
0    1.0
1    NaN
2    2.0
3    NaN
4    0.0
5    3.0
dtype: float64
In [44]:

# 传入的参数是字典
data.replace({-999:np.nan,-1000:0})
Out[44]:
0    1.0
1    NaN
2    2.0
3    NaN
4    0.0
5    3.0
dtype: float64
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
重命名轴索引

data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
tranform = lambda x: x[:4].upper()
data.index.map(tranform)
>>>
Index(['OHIO', 'COLO', 'NEW '], dtype='object')
In [47]:

data
Out[47]:
one	two	three	four
Ohio	0	1	2	3
Colorado	4	5	6	7
New York	8	9	10	11
In [49]:

data.index = data.index.map(tranform)
data
Out[49]:
one	two	three	four
OHIO	0	1	2	3
COLO	4	5	6	7
NEW	8	9	10	11
In [50]:

data.rename(index = str.title,columns = str.upper)
Out[50]:
ONE	TWO	THREE	FOUR
Ohio	0	1	2	3
Colo	4	5	6	7
New	8	9	10	11
In [53]:

# 对部分轴标签更新
data.rename(index = {'OHIO':'INDIANA'},columns = {'three':'peekaboo'})
Out[53]:
one	two	peekaboo	four
INDIANA	0	1	2	3
COLO	4	5	6	7
NEW	8	9	10	11
In [55]:

# 修改源数据集
data.rename(index = {'OHIO':'INDIANA'},inplace = True)
data
Out[55]:
one	two	three	four
INDIANA	0	1	2	3
COLO	4	5	6	7
NEW	8	9	10	11
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
离散化和面元划分

# cut函数划分
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages,bins)
cats
>>>
[(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
Length: 12
Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]
In [58]:

cats.codes
Out[58]:
array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)
In [59]:

cats.categories
Out[59]:
IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]],
              closed='right',
              dtype='interval[int64]')
In [60]:

pd.value_counts(cats)
Out[60]:
(18, 25]     5
(35, 60]     3
(25, 35]     3
(60, 100]    1
dtype: int64
In [62]:

# 设置闭端，默认为右闭
pd.cut(ages,[18,26,36,61,100],right = False)
Out[62]:
[[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
Length: 12
Categories (4, interval[int64]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]
In [64]:

# 设置面元名称
pd.cut(ages,bins,labels = ['Youth', 'YoungAdult', 'MiddleAged','Senior'])
Out[64]:
[Youth, Youth, Youth, YoungAdult, Youth, ..., YoungAdult, Senior, MiddleAged, MiddleAged, YoungAdult]
Length: 12
Categories (4, object): [Youth < YoungAdult < MiddleAged < Senior]
In [66]:

# 将均匀分布的数据分组
data = np.random.rand(20)
pd.cut(data,4,precision =2)  # precision=2，限定小数只有两位
Out[66]:
[(0.016, 0.25], (0.71, 0.94], (0.016, 0.25], (0.016, 0.25], (0.016, 0.25], ..., (0.25, 0.48], (0.016, 0.25], (0.25, 0.48], (0.016, 0.25], (0.48, 0.71]]
Length: 20
Categories (4, interval[float64]): [(0.016, 0.25] < (0.25, 0.48] < (0.48, 0.71] < (0.71, 0.94]]
In [67]:

# qcut函数使用样本分位数的到大小相等的面元
data = np.random.randn(1000)
cats = pd.qcut(data,4)
cats
Out[67]:
[(-0.704, -0.0111], (-0.704, -0.0111], (-3.468, -0.704], (0.623, 2.937], (-3.468, -0.704], ..., (-0.0111, 0.623], (0.623, 2.937], (0.623, 2.937], (-0.0111, 0.623], (-3.468, -0.704]]
Length: 1000
Categories (4, interval[float64]): [(-3.468, -0.704] < (-0.704, -0.0111] < (-0.0111, 0.623] < (0.623, 2.937]]
In [68]:

cats.value_counts()
Out[68]:
(-3.468, -0.704]     250
(-0.704, -0.0111]    250
(-0.0111, 0.623]     250
(0.623, 2.937]       250
dtype: int64
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
检测和过滤异常值

data = pd.DataFrame(np.random.randn(1000,4))
data.describe()
>>>

0	1	2	3
count	1000.000000	1000.000000	1000.000000	1000.000000
mean	0.002067	-0.003415	0.041089	-0.016073
std	1.031838	1.019784	0.997811	0.994850
min	-2.968172	-3.180084	-3.567853	-2.849138
25%	-0.701257	-0.673598	-0.601548	-0.644626
50%	-0.040249	-0.015020	0.091156	-0.000759
75%	0.703070	0.655398	0.677955	0.628997
max	3.869659	3.395740	3.114127	3.159251
In [71]:

col = data[2]
col[np.abs(col) > 3]  #选取绝对值大于3
Out[71]:
282    3.114127
362   -3.567853
470   -3.228497
988   -3.413098
Name: 2, dtype: float64
In [72]:

data[(np.abs(data) > 3).any(1)]  # 选取绝对值大于3的全部行
Out[72]:
0	1	2	3
77	1.912827	0.276372	0.251493	3.159251
112	0.009310	-3.119472	-0.054482	-0.544765
182	3.037274	-0.567013	-1.022662	-0.685939
245	3.122183	-2.466421	0.644751	-0.467498
282	-0.628889	-0.719120	3.114127	1.266619
321	1.316235	3.395740	0.674903	0.378740
362	0.850918	0.224832	-3.567853	-0.151309
399	3.243082	0.196252	0.229971	-0.268654
449	1.048033	-3.020496	0.314366	0.164850
453	3.131162	-0.212424	-1.281723	0.429625
470	0.508345	1.827191	-3.228497	-0.169183
822	0.518345	1.088880	1.582630	3.024338
870	3.869659	1.353480	-1.020291	0.029672
921	-1.229088	-3.180084	-1.488577	0.989180
965	3.222980	-1.912414	-0.121301	-0.005904
988	-0.234364	0.253744	-3.413098	0.340663
In [74]:

data[np.abs(data) > 3] = np.sign(data) *3  # 将值限制在-3到3区间内
data.describe() 
Out[74]:
0	1	2	3
count	1000.000000	1000.000000	1000.000000	1000.000000
mean	0.000441	-0.003490	0.042185	-0.016257
std	1.026658	1.017574	0.993493	0.994279
min	-2.968172	-3.000000	-3.000000	-2.849138
25%	-0.701257	-0.673598	-0.601548	-0.644626
50%	-0.040249	-0.015020	0.091156	-0.000759
75%	0.703070	0.655398	0.677955	0.628997
max	3.000000	3.000000	3.000000	3.000000
In [75]:

np.sign(data).head() #np.sign(data)生成-1和1
Out[75]:
0	1	2	3
0	-1.0	-1.0	1.0	1.0
1	1.0	1.0	-1.0	1.0
2	1.0	-1.0	1.0	1.0
3	1.0	1.0	1.0	1.0
4	1.0	-1.0	1.0	-1.0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
排列和随机采样

# numpy.random.permutation函数进行排序
df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
df
>>>

0	1	2	3
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15
4	16	17	18	19
In [77]:

sampler = np.random.permutation(5)
sampler
Out[77]:
array([2, 0, 4, 3, 1])
In [78]:

df.take(sampler)
Out[78]:
0	1	2	3
2	8	9	10	11
0	0	1	2	3
4	16	17	18	19
3	12	13	14	15
1	4	5	6	7
In [79]:

# 选取随机子集
df.sample(n= 3)
Out[79]:
0	1	2	3
2	8	9	10	11
4	16	17	18	19
3	12	13	14	15
In [80]:

# 替换的方式产生样本
choices = pd.Series([5,7,-1,6,4])
draws = choices.sample(n = 10,replace = True)
draws
Out[80]:
2   -1
1    7
1    7
0    5
2   -1
3    6
2   -1
3    6
4    4
3    6
dtype: int64
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
字符串操作

字符串对象方法

# split拆分
val = 'a,b,   guido'
val.split(',')
>>>
['a', 'b', '   guido']
In [83]:

# split与strip一起使用，去除空白符和换行符
pieces = [x.strip() for x in val.split(',')]
pieces
Out[83]:
['a', 'b', 'guido']
In [84]:

# 字符串定位
'guido' in val
Out[84]:
True
In [85]:

val.index(',')
Out[85]:
1
In [86]:

val.find(':')
Out[86]:
-1
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 注意 find 和 index 的区别:如果找不到字符串，index 将会引发一个异常(而 不是返回-1)
# 返回指定字串出现次数
val.count(',')
>>>
2
In [88]:

# replace用于替换,传入空字符串可用作删除
val.replace(',' , '::')
Out[88]:
'a::b::   guido'
In [89]:

val.replace(',' , '')
Out[89]:
'ab   guido'
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
python内置的字符串方法：

pandas字符串方法：


--------------------- 
版权声明：本文为CSDN博主「luckygirk」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/luckygirk/article/details/99488033
