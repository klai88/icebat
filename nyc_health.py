import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pygeocode import geocoder
from ggplot import *
from sklearn import cluster, linear_model, tree,feature_selection, decomposition, metrics, naive_bayes, cross_validation, neighbors
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import scale
from mpltools import style

restaurants = pd.read_csv('restaurants.csv')
restaurants = restaurants.dropna()
food_inspections = pd.read_csv('food_inspections.csv')

#Exploratory data/graphing
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#PLOTS
style.use('ggplot')
ggplot(aes(x='SCORE'),data=food_inspections)+geom_bar()+xlim(0,35)+ggtitle('Number of Inspections Receving Scores')


graded = food_inspections[food_inspections['CURRENTGRADE'] != 'Z']
graded = graded[graded['CURRENTGRADE'] != 'P']
ggplot(aes(x='CURRENTGRADE'),data=graded)+geom_bar()+ggtitle('Number of Inspections Receiving Each Grade')



#scatter matrix for flags/critical flags
pd.tools.plotting.scatter_matrix(restaurants[['flags_per_insp', 'crit_flags_per_insp','SCORE']], alpha=0.2, diagonal='hist')
#scatter matrix for rates violations
pd.tools.plotting.scatter_matrix(restaurants[['02_per_insp', '04_per_insp', '06_per_insp', '08_per_insp', '10_per_insp', 'SCORE']], alpha=0.2, diagonal='hist')

#cuisine graph
ggplot(aes(x='CUISINECODE', y='SCORE'), data=restaurants[['CUISINECODE', 'SCORE']].groupby(['CUISINECODE'], as_index=False).mean())+geom_line()+ggtitle('Avg Score by Cuisine')

#ggplot(aes(x=range(len(restaurants.ZIPCODE.unique())), y='SCORE'), data=restaurants[['ZIPCODE', 'SCORE']].groupby(['ZIPCODE'], as_index=False).mean())+geom_line()+ggtitle('Avg Score by Zip Code')

# plt.show()

#percentage of grades
ggplot(aes(x='CURRENTGRADE'), data = food_inspections)+geom_hist()

restaurants[['UP_DOWN', 'ORIGINAL_GRADE']].groupby(['ORIGINAL_GRADE', 'UP_DOWN']).count()
restaurants[['NEW_GRADE', 'ORIGINAL_GRADE']].groupby(['ORIGINAL_GRADE', 'NEW_GRADE']).count()

#Classification
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

# #use only inspections where
# restaurants = restaurants[restaurants['FIRST_INSP'] < '2012']
'02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp', 'NEWCODE', 'Manhattan'

'02_per_insp', '06_per_insp', '10_per_insp', '09_per_insp', '04_per_insp', '08_per_insp', '03_per_insp', 
'05_per_insp', '22_per_insp', '99_per_insp', '16_per_insp', '15_per_insp', '20_per_insp', '18_per_insp', 
'07_per_insp', '12_per_insp', 'flags_per_insp', 'crit_flags_per_insp', 'NEWCODE', 'Manhattan',
'Brooklyn', 'Bronx', 'Queens', 'Staten Island'



#Given rates of violations, predict an average score
input = restaurants[['02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp','Manhattan', 'Bronx',
'Brooklyn', 'Queens', 'Staten Island',
'cuisine_1', 'cuisine_2',  'cuisine_3','cuisine_4', 'cuisine_5', 'cuisine_6']].values
output = restaurants['AVG_Grade'].values
x_train, x_test, y_train, y_test = cross_validation.train_test_split(input, output, test_size=.3, random_state=123)


multi = naive_bayes.GaussianNB()
multi.fit(x_train, y_train)
feature_selection.univariate_selection.f_regression(x_train, y_train)
#p-values show Manhattan, Bronx, Brooklyn, cuisine_1 are insignificant, but taking them out doesn't add much to score
multi.score(x_test, y_test)

print metrics.classification_report(y_train, multi.predict(x_train))
metrics.confusion_matrix(multi.predict(x_train), y_train)

print metrics.classification_report(y_test, multi.predict(x_test))
print metrics.confusion_matrix(multi.predict(x_test), y_test)

######OVERALL#######
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##Given rates of violations, predict if restaurant score goes up or down

input = restaurants[['02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp','Manhattan', 'Bronx',
'Brooklyn', 'Queens', 'Staten Island',
'cuisine_1', 'cuisine_2',  'cuisine_3','cuisine_4', 'cuisine_5', 'cuisine_6']].values
output = restaurants['UP_DOWN'].values
input_norm = scale(input)
input_norm = np.append(input_norm, restaurants[['NEWCODE', 'Manhattan']].values, axis=1)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(input, output, test_size=.3, random_state=123)

multi = naive_bayes.GaussianNB()
multi.fit(x_train, y_train)
feature_selection.univariate_selection.f_regression(x_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, multi.predict(x_train), pos_label=1)

metrics.auc(fpr,tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, multi.predict(x_test), pos_label=1)

metrics.auc(fpr,tpr)

print metrics.classification_report(y_train, multi.predict(x_train))
metrics.confusion_matrix(multi.predict(x_train), y_train)

print metrics.classification_report(y_test, multi.predict(x_test))
print metrics.confusion_matrix(multi.predict(x_test), y_test)

######GRADE C#######
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#three ways
grade_C = restaurants[restaurants['ORIGINAL_SCORE'] >27]
grade_C[['SCORE', 'TOTAL']].groupby(['TOTAL'], as_index=False).mean()
grade_C[['TOTAL']].groupby(['TOTAL']).count()
grade_C[['UP_DOWN']].groupby(['UP_DOWN']).count()
#About the same


input = grade_C[['02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp','Manhattan', 'Bronx',
'Brooklyn', 'Queens', 'Staten Island',
'cuisine_1', 'cuisine_2',  'cuisine_3','cuisine_4', 'cuisine_5', 'cuisine_6']].values
output = grade_C['UP_DOWN'].values
input_norm = scale(input)
input_norm = np.append(input_norm, grade_C[['NEWCODE', 'Manhattan']].values, axis=1)
# '02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
# 

x_train, x_test, y_train, y_test = cross_validation.train_test_split(input, output, test_size=.3, random_state=123)


multi = naive_bayes.GaussianNB()
multi.fit(x_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, multi.predict(x_train), pos_label=1)
metrics.auc(fpr,tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, multi.predict(x_test), pos_label=1)
metrics.auc(fpr,tpr)

print metrics.classification_report(y_train, multi.predict(x_train))
metrics.confusion_matrix(multi.predict(x_train), y_train)

print metrics.classification_report(y_test, multi.predict(x_test))
metrics.confusion_matrix(multi.predict(x_test), y_test)


######GRADE B#######
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#Original score of grade B
grade_B = restaurants[restaurants['ORIGINAL_SCORE'] <28]
grade_B = grade_B[grade_B['ORIGINAL_SCORE'] > 13]
grade_B[['SCORE', 'TOTAL']].groupby(['TOTAL'], as_index=False).mean()
grade_B[['TOTAL']].groupby(['TOTAL']).count()
grade_B[['UP_DOWN']].groupby(['UP_DOWN']).count()
#About the same

input = grade_B[['02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp','Manhattan', 'Bronx',
'Brooklyn', 'Queens', 'Staten Island',
'cuisine_1', 'cuisine_2',  'cuisine_3','cuisine_4', 'cuisine_5', 'cuisine_6']].values
output = grade_B['UP_DOWN'].values
input_norm = scale(input)
input_norm = np.append(input_norm, grade_B[['NEWCODE', 'Manhattan']].values, axis=1)
# '02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
# 

x_train, x_test, y_train, y_test = cross_validation.train_test_split(input, output, test_size=.3, random_state=123)


multi = naive_bayes.GaussianNB()
multi.fit(x_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, multi.predict(x_train), pos_label=1)
metrics.auc(fpr,tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, multi.predict(x_test), pos_label=1)
metrics.auc(fpr,tpr)

print metrics.classification_report(y_train, multi.predict(x_train))
metrics.confusion_matrix(multi.predict(x_train), y_train)

print metrics.classification_report(y_test, multi.predict(x_test))
metrics.confusion_matrix(multi.predict(x_test), y_test)

######GRADE A#######
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#looks good for a_plus - UP_DOWN
grade_A = restaurants[restaurants['ORIGINAL_SCORE'] < 14]
grade_A[['SCORE', 'TOTAL']].groupby(['TOTAL'], as_index=False).mean()
grade_A[['TOTAL']].groupby(['TOTAL']).count()
grade_A[['UP_DOWN']].groupby(['UP_DOWN']).count()
#a plus flipped using the different systems. 
#TOTAL system was being too harsh on grade_A, as any POSITIVE made it a zero

input = grade_A[['02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 
'flags_per_insp', 'crit_flags_per_insp','Manhattan', 'Bronx',
'Brooklyn', 'Queens', 'Staten Island',
'cuisine_1', 'cuisine_2',  'cuisine_3','cuisine_4', 'cuisine_5', 'cuisine_6']].values
output = grade_A['UP_DOWN'].values
input_norm = scale(input)
input_norm = np.append(input_norm, grade_A[['NEWCODE', 'Manhattan']].values, axis=1)
# '02_per_insp', '06_per_insp', '10_per_insp', '04_per_insp', '08_per_insp', 


x_train, x_test, y_train, y_test = cross_validation.train_test_split(input, output, test_size=.3, random_state=123)
multi = naive_bayes.GaussianNB()
multi.fit(x_train, y_train)

fpr, tpr, thresholds = metrics.roc_curve(y_train, multi.predict(x_train), pos_label=1)
metrics.auc(fpr,tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, multi.predict(x_test), pos_label=1)
metrics.auc(fpr,tpr)

print metrics.classification_report(y_train, multi.predict(x_train))
metrics.confusion_matrix(multi.predict(x_train), y_train)

print metrics.classification_report(y_test, multi.predict(x_test))
metrics.confusion_matrix(multi.predict(x_test), y_test)




####Create Neighborhoods

def Neighborhoods(x):
	

