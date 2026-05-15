import pandas as pd


college = pd.read_csv("College.csv")


college2 = pd.read_csv('College.csv', index_col=0)
college3 = college.rename({'Unnamed: 0': 'College'},axis=1)
college3 = college3.set_index('College')

# print(college)
# print(college2)
# print(college3)

college = college3

# print(college.describe())

pd.plotting.scatter_matrix(college[['Top10perc', 'Apps', 'Enroll']])

college.boxplot(column='Outstate', by='Private')

college['Top10perc']

college['Elite'] = pd.cut(college['Top10perc'] > 50, [-1,0.5,1.5], labels=['No', 'Yes'])

college['Elite']

college['Elite'].value_counts()

college.boxplot(column='Outstate', by='Elite')

# Part G:
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
college.hist(column='Outstate', ax=axes[0,0], bins=20)
college.hist(column='Room.Board', ax=axes[0,1], bins=30)
college.hist(column='Books', ax=axes[1,0], bins=13)
college.hist(column='Personal', ax=axes[1,1], bins=23)
college.hist(column='PhD', ax=axes[2,0], bins=16)
college.hist(column='Terminal', ax=axes[2,1], bins=17)
college.hist(column='S.F.Ratio', ax=axes[0,2], bins=20)
college.hist(column='perc.alumni', ax=axes[1,2], bins=20)
college.hist(column='Expend', ax=axes[2,2], bins=20)

# Part H:
college.boxplot(column='PhD', by='Elite')

college.boxplot(column='Terminal', by='Elite')

pd.plotting.scatter_matrix(college[['Outstate', 'PhD', 'Terminal']])


#Problem 9
#Part A:
auto = pd.read_csv('Auto.csv')
auto = auto.dropna()
auto.head()

dtypes = auto.dtypes
quantitative = dtypes[dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))].index.tolist()
qualitative = dtypes[dtypes.apply(lambda x: pd.api.types.is_object_dtype(x))].index.tolist()
print("Qualitative: ", qualitative)
print("Quantitative: ", quantitative)

# Part B:
import numpy as np
for i in quantitative:
    print(i, ":", np.min(auto[i]), np.max(auto[i]))

# Part C:
for i in quantitative:
    print(i, ":", np.mean(auto[i]), np.std(auto[i]))

#Part D:
auto_subset = auto.drop(auto.index[9:85])
for i in quantitative:
    print(i, ":", np.min(auto_subset[i]), np.max(auto_subset[i]), np.mean(auto_subset[i]), np.std(auto_subset[i]))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
auto.hist(column='mpg', ax=axes[0,0], bins=20)
auto.hist(column='cylinders', ax=axes[0,1], bins=30)
auto.hist(column='displacement', ax=axes[1,0], bins=13)
auto.hist(column='weight', ax=axes[1,1], bins=16)

auto.plot.scatter(x='mpg', y='cylinders')
auto.plot.scatter(x='mpg', y='displacement')
auto.plot.scatter(x='mpg', y='horsepower')
auto.plot.scatter(x='mpg', y='weight')
auto.plot.scatter(x='mpg', y='acceleration')

pd.plotting.scatter_matrix(auto[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration']])

# Chapter 3
# Problem 13
# Part A:
np.random.seed(1)
x = np.random.normal(0, 1, 100)
plt.hist(x, bins=20)

#Part B:
eps = np.random.normal(0, 0.25, 100)
plt.hist(eps, bins=20)

# Part C:
y = -1 + 0.5*x + eps
print(len(y))
print('b1', np.corrcoef(x, y)[0,1])
print('b0', np.mean(y) - np.mean(x)*np.corrcoef(x, y)[0,1])

# Part D:
plt.scatter(x, y)

# Part E:
import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

b1 = model.params[1]
b0 = model.params[0]
print('b1', b1)
print('b0', b0)

#Part F:
plt.scatter(x[:,1], y)
plt.plot(x[:,1], b0 + b1*x[:,1], 'r')
plt.plot(x[:,1], -1 + 0.5*x[:,1], 'g')
plt.legend(['OLS', 'Population'])

#Part G:
x2 = x[:,1]**2
x2 = x2.reshape(-1,1)
x2 = sm.add_constant(x2)
x2 = np.concatenate((x, x2), axis=1)
model = sm.OLS(y, x2).fit()
model.summary()

b1 = model.params[1]
b2 = model.params[2]
b0 = model.params[0]
print('b1', b1)
print('b2', b2)
print('b0', b0)

plt.scatter(x[:,1], y)
plt.plot(x[:,1], b0 + b1*x[:,1] + b2*x[:,1]**2, 'r') # why are you ugly?
plt.plot(x[:,1], -1 + 0.5*x[:,1], 'g')
plt.legend(['OLS', 'Population'])