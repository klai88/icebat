#directory for downloads/dohmh folder

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pygeocode import geocoder
from ggplot import *
from sklearn import cluster, linear_model, feature_selection, decomposition, metrics, naive_bayes, cross_validation
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import scale

#Data Mining
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
food = pd.read_csv('WebExtract.txt')
food.head()
del food['RECORDDATE']

action = pd.read_csv('Action.txt')
cuisine = pd.read_csv('Cuisine.txt')
violation = pd.read_csv('Violation.csv')
violation = violation.drop_duplicates('VIOLCODE', take_last=True)
violation['VIOLGROUP'] = [violation['VIOLCODE'][x][:2] for x in violation.index]
del violation['STARTDATE']
del violation ['ENDDATE']
#violation.to_csv('violation_clean.csv')

#Violation groups
# 1 - Technicality
# 2 - Food cooking
# 3 - Raw food handling
# 4 - Food worker prep cleanliness/contamination/food hazards - worker's fault
# 4 also includes flies/mice/roaches/live animal contamination or evidence of - worker's fault
# 5 - sewage disposal, bathrooms/facilities to clean
# 6 - storage of food and utensils not clean, personal cleanliness, contamination from workers
# 7 - generic
######non critical point
# 8 - not critical cleanliness
# 9 - food procedures thawing, food contact
#10 - bathroom, facilities


len(food[food['VIOLCODE'] == '04M'])

#group the violation numbers to be a violation group


food = pd.merge(food, cuisine, on='CUISINECODE')
food = pd.merge(food, violation, on='VIOLCODE')
#order by restaurant id and inspection date
food = food.sort_index(by = ['CAMIS', 'INSPDATE', 'SCORE'], ascending= [True, True, True])

#Histograms
ggplot(aes(x='SCORE'),data=food_inspections)+geom_hist(binwidth=5)


ggplot(aes(x='FLAGS', y='SCORE'),data=restaurants)+geom_point()+ facet_wrap("BORO")
h = food['SCORE'].hist(bins=100)
plt.title('Histogram of Health Scores')

#clustered around the 0-40 range

#Might need to change rows for scores and grade purposes grouped by gradedate, CAMIS
# food_inspections = food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'INSPDATE', 'SCORE', 'CURRENTGRADE', 'GRADEDATE']]
# food_inspections = food_inspections.drop_duplicates()

# food_grades = food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'GRADEDATE', 'CURRENTGRADE', 'SCORE']]
# food_grades = food_grades.drop_duplicates()



"""Each line isn't one inspection, but if one inspection has multiple violations then an inspection will have
more than one row"""

#use food_inspection
#create a colummn for each violation group for each inspection to see which categories they violated
for elem in food.VIOLGROUP.unique():
	food[str(elem)] = [1 if food['VIOLGROUP'][x] == elem else 0 for x in food.index]

list(food.columns)
food_inspections = food[['CAMIS', 'INSPDATE', '02', '06', '10', '09', '04', '08', '03', '05', '22', '99', '16', 
'15', '20', '18', '07', '12']]

food_inspections = food_inspections.groupby(['CAMIS','INSPDATE'], as_index=False).sum()

#check data is correct
food[food['CAMIS']==30075445]

food_inspections[food_inspections['CAMIS']==30075445]

#merge back other data 
food_inspections = pd.merge(food_inspections, food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 
	'PHONE', 'INSPDATE', 'ACTION','SCORE','CURRENTGRADE', 'GRADEDATE', 'CODEDESC']], on=['CAMIS', 'INSPDATE'], how='inner')
food_inspections = food_inspections.drop_duplicates(['CAMIS', 'INSPDATE'])
#food_inspections.to_csv('food_inspections.csv')

"""probability of violation X GIVEN avg score, rate of all violations"""

"""sometimes inspections did not give scores"""


food_inspections['SCORE'].groupby(food_inspections['BORO']).agg([np.mean, np.std])
#the lower the score, the better the grade
#There's a grade P

food_inspections['SCORE'].groupby(food_inspections['CURRENTGRADE']).agg([np.mean, np.std])
#what boro is the worst?


###############RESTAURANTS PER ROW
#list of all violations groups that a restaurant had
restaurants = food[['CAMIS', '02', '06', '10', '09', '04', '08', '03', '05', '22', '99', '16', 
'15', '20', '18', '07', '12']]
restaurants = restaurants.groupby(['CAMIS'], as_index = False).sum()
food = food.sort_index(by = ['CAMIS', 'INSPDATE'], ascending= [True, False])
restaurants = pd.merge(restaurants, food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'CURRENTGRADE', 
'CODEDESC']], how='inner')
restaurants = restaurants.drop_duplicates(['CAMIS'])

scores = food_inspections[['CAMIS', 'SCORE']]
scores = scores.groupby(['CAMIS'], as_index=False).mean()
restaurants = pd.merge(restaurants, scores, how ='inner')

critical = food[['CAMIS', 'CRITICALFLAG']]
critical['CRITICALFLAG'] = [1 if critical['CRITICALFLAG'][x] == 'Y' else 0 for x in critical.index]
critical = critical.groupby(['CAMIS'], as_index = False).sum()
restaurants = pd.merge(restaurants, critical,how ='inner')

flags = food[['CAMIS','CRITICALFLAG']]
flags['FLAGS'] = [1 for x in flags.index]
del flags['CRITICALFLAG']
flags = flags.groupby(['CAMIS'], as_index=False).sum()

restaurants = pd.merge(restaurants, flags, how ='inner')

inspection = food_inspections[['CAMIS', 'INSPDATE']]
inspection['INSPECTIONS'] = [1 for x in inspection.index]
del inspection['INSPDATE']
inspection = inspection.groupby(['CAMIS'], as_index=False).sum()

restaurants = pd.merge(restaurants, inspection, how='inner')

#give rate of violation per inspection

elements = ['02', '06', '10', '09', '04', '08', '03', '05', '22', '99', '16', 
'15', '20', '18', '07', '12']

for elem in elements:
	restaurants[elem+'_per_insp'] = [float(restaurants[elem][x])/inspection['INSPECTIONS'][x] for x in restaurants.index]

def Boro(x):
	if x == 1:
		return 'Manhattan'
	if x == 2:
		return 'Bronx'
	if x == 3:
		return 'Brooklyn'
	if x == 4:
		return 'Queens'
	if x == 5:
		return 'Staten Island'

restaurants['BORO_name'] = [Boro(restaurants['BORO'][x]) for x in restaurants.index]
for elem in restaurants.BORO_name.unique():
	restaurants[str(elem)] = [1 if restaurants['BORO_name'][x] == elem else 0 for x in restaurants.index]

restaurants = restaurants.dropna()
#flags per inspection for each restaurant
restaurants['flags_per_insp'] = [float(restaurants['FLAGS'][x])/restaurants['INSPECTIONS'][x] for x in restaurants.index]
restaurants['crit_flags_per_insp'] = [float(restaurants['CRITICALFLAG'][x])/restaurants['INSPECTIONS'][x] for x in restaurants.index]


restaurants['flags_per_insp'].groupby(restaurants['BORO']).agg([np.mean, np.std])
restaurants['crit_flags_per_insp'].groupby(restaurants['BORO']).agg([np.mean, np.std])
#these are not varying by boro, which is good

#average score per restaurant grouped by each restaurant's most recent grade
restaurants['SCORE'].groupby(restaurants['CURRENTGRADE']).agg([np.mean, np.std])

#create a 'avg grade' based on the average score of each restaurant
#Create 'Grade' based on the average score
restaurants['AVG_Grade'] = [1 if restaurants['SCORE'][x] < 14 else 2 if restaurants['SCORE'][x] > 14 and restaurants['SCORE'][x] < 28 else 3 for x in restaurants.index]

restaurants['GRADENAME'] = ['A' if restaurants['SCORE'][x] < 14 else 'B' if restaurants['SCORE'][x] > 14 and restaurants['SCORE'][x] < 28 else 'C' for x in restaurants.index]

restaurants = pd.merge(restaurants, cuisine[['CUISINECODE', 'NEWCODE']], on ='CUISINECODE')
#delete restaurants with a zero, which is 'N/A'
restaurants = restaurants[restaurants['CUISINECODE'] != 0]

food_inspections = pd.merge(food_inspections,cuisine, on ='CODEDESC')
food_inspections = food_inspections[food_inspections['CUISINECODE'] != 0]

#make a few metrics on the ups and downs of a restaurant's score over time.

#use restaurants with at least 5 inspections

#standard dev - no
#variance - no
#number of ups and downs

scores = food_inspections[['CAMIS', 'SCORE']]

restaurants = restaurants[restaurants['ZIPCODE']!='N/A']
restaurants = restaurants[restaurants['ZIPCODE']!='nan']

food_inspections[food_inspections['CAMIS']==41667469]['SCORE'][1:2].values-food_inspections[food_inspections['CAMIS']==41667469]['SCORE'][0:1].values

test = pd.DataFrame(columns=('CAMIS', 'ORIGINAL_SCORE', 'VARY'))
counting = pd.DataFrame(columns=('CAMIS', 'COUNTING'))
total = pd.DataFrame(columns=('CAMIS', 'TOTAL'))
new_grade = pd.DataFrame(columns = ('CAMIS', 'UP_DOWN'))
newer_grade = pd.DataFrame(columns = ('CAMIS', 'NEW_GRADE'))

totals=[]
testy=[]
#On columns COUNTING and TOTAL, 1 is score got lower, 0 is score went up (on average)
for elem in food_inspections.CAMIS.unique():
	scores = food_inspections[food_inspections['CAMIS'] == elem]['SCORE']
	original_score = scores[0:1].values
	scores=scores.dropna()
	n = len(scores)
	cost = []
	for i in range(n):
		if i>0:
			score = scores[i:i+1].values - scores[i-1:i].values
			cost.append(score)
	count = sum([cost[i] for i in range(len(cost))])
	new_score = original_score+count
	totals.append(count)
	rows = pd.DataFrame({'CAMIS': [elem], 'ORIGINAL_SCORE': [original_score], 'VARY': [count]})
	test = test.append(rows)
	if count > n/2:
		costs = 1
		counts = pd.DataFrame({'CAMIS': [elem], 'COUNTING': [costs]})
		counting = counting.append(counts)
	else: 
		costs = 0
		counts = pd.DataFrame({'CAMIS': [elem], 'COUNTING': [costs]})
		counting = counting.append(counts)
	if sum(cost) < 0:
		costs_total = 1
		counts = pd.DataFrame({'CAMIS': [elem], 'TOTAL': [costs_total]})
		total = total.append(counts)
	else: 
		costs_total = 0
		counts = pd.DataFrame({'CAMIS': [elem], 'TOTAL': [costs_total]})
		total = total.append(counts)
	if new_score < 14 and original_score < 14:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	elif new_score > 13 and original_score < 14:
		up_down = 0
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	elif original_score > 13 and original_score < 28 and new_score >27:
		up_down = 0
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	elif original_score > 13 and original_score < 28 and new_score < 28:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	elif original_score > 27 and new_score > 27:
		up_down = 0
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	elif original_score > 27 and new_score < 28:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'UP_DOWN': [up_down]})
		new_grade = new_grade.append(new_row)
	if original_score < 14 and new_score < 14:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score < 14 and new_score > 13 and new_score < 28:
		up_down = 2
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score < 14 and new_score > 27:
		up_down = 3
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 13 and original_score < 28 and new_score >27:
		up_down = 3
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 13 and original_score < 28 and new_score < 28 and new_score > 13:
		up_down = 2
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 13 and original_score < 28 and new_score < 14:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 27 and new_score > 27:
		up_down = 3
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 27 and new_score < 28 and new_score > 13:
		up_down = 2
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)
	elif original_score > 27 and new_score < 14:
		up_down = 1
		new_row = pd.DataFrame({'CAMIS': [elem], 'NEW_GRADE': [up_down]})
		newer_grade = newer_grade.append(new_row)

del restaurants['UP_DOWN']
del restaurants['NEW_GRADE']
restaurants = pd.merge(restaurants, counting)
restaurants = pd.merge(restaurants, total)
restaurants = pd.merge(restaurants, new_grade)
restaurants = pd.merge(restaurants, newer_grade)

#Where the restaurant started
original_score = food_inspections[['CAMIS', 'SCORE']].drop_duplicates(cols = 'CAMIS')
original_score.columns=['CAMIS', 'ORIGINAL_SCORE']
restaurants = pd.merge(restaurants, original_score)


first_insp = food_inspections[['CAMIS', 'INSPDATE']].drop_duplicates(cols='CAMIS')
first_insp.columns=['CAMIS', 'FIRST_INSP']
first_insp['FIRST_INSP'] = [first_insp['FIRST_INSP'][i][0:4] for i in first_insp.index]
restaurants = pd.merge(restaurants, first_insp)

restaurants['ORIGINAL_GRADE'] = ['A' if restaurants['ORIGINAL_SCORE'][x] < 14 else 'B' if restaurants['ORIGINAL_SCORE'][x] > 14 and restaurants['ORIGINAL_SCORE'][x] < 28 else 'C' for x in restaurants.index]

elements = [6, 3, 1, 2, 4, 5]

for elem in elements:
	restaurants['cuisine_'+str(elem)] = [1 if restaurants['NEWCODE'][x] == elem else 0 for x in restaurants.index]


food_inspections.to_csv('food_inspections.csv', index=False)
restaurants.to_csv('restaurants.csv', index=False)
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################