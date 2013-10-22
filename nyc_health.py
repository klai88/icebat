#directory for downloads/dohmh folder

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pygeocode import geocoder
from ggplot import *
from sklearn import cluster, linear_model, feature_selection

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
#10 - 


len(food[food['VIOLCODE'] == '04M'])

#group the violation numbers to be a violation group


food = pd.merge(food, cuisine, on='CUISINECODE')
food = pd.merge(food, violation, on='VIOLCODE')
#order by restaurant id and inspection date
food = food.sort_index(by = ['CAMIS', 'INSPDATE', 'SCORE'], ascending= [True, True, True])
len(food[food['VIOLGROUP'] == '07'])

ggplot(aes(x='SCORE'),data=food)+geom_hist()

h = food['SCORE'].hist(bins=100)
plt.title('Histogram of Health Scores')

#clustered around the 0-40 range

#Might need to change rows for scores and grade purposes grouped by gradedate, CAMIS
# food_inspections = food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'INSPDATE', 'SCORE', 'CURRENTGRADE', 'GRADEDATE']]
# food_inspections = food_inspections.drop_duplicates()

# food_grades = food[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'GRADEDATE', 'CURRENTGRADE', 'SCORE']]
# food_grades = food_grades.drop_duplicates()



"""Each line isn't one inspection, but if one inspection has multiple violations then an inspection will have
more than one row
"""

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


food_inspections['SCORE'].groupby(food_inspections['BORO']).agg([np.mean, np.std])
#the lower the score, the better the grade
#There's a grade P

food_inspections['SCORE'].groupby(food_inspections['CURRENTGRADE']).agg([np.mean, np.std])
#what boro is the worst?

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

#given a score, location, cuisine predict the violations?

#show clusters of historical violations - who are the perpetrators?
#sum of violations

#indicator of a violator of the group or now
def Violate(x):
	if x > 0:
		return 1
	else:
		return 0
rest_single = restaurants.copy()


for elem in features:
	rest_single[elem] = [Violate(rest_single[elem][x]) for x in rest_single.index]

features = ['02', '06', '10', '09', '04', '08', '03', '05', '22', '99', '16', 
'15', '20', '18', '07', '12']


#k-means
clf = cluster.k_means(restaurants[features].values,3)
restaurants['clusters'] = clf[1]

#plt.scatter(restaurants['02'].values, restaurants['06'].values, c=list(restaurants.clusters.values))

plt.scatter(restaurants['FLAGS'].values, restaurants['SCORE'].values, c=list(restaurants.clusters.values))
plt.title('Clusters Identifed by color')
plt.show()
#cluster based on the violation group
#same neighborhoods? same cuisine? probably not.


######################
#bayesian probability
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

restaurants['BORO'] = [Boro(restaurants['BORO'][x]) for x in restaurants.index]

restaurants.CODEDESC.unique()
#Linear regress
for elem in restaurants.BORO.unique():
	restaurants[str(elem)] = restaurants['BORO'] == elem

restaurants = restaurants.dropna()

features = ['02', '06', '10', '09', '04', '08', '03', '05', '22', '99', '16', 
'15', '20', '18', '07', '12', 'Bronx', 'Brooklyn', 'Manhattan','Queens', 'Staten Island']
#flags per inspection, critical per inspection
input = restaurants[features].values
output = restaurants[['SCORE']].values

lm = linear_model.LinearRegression()
lm.fit(input,output)
feature_selection.univariate_selection.f_regression(input,output)
lm.score(input,output)


#address lookup for longitude and latitude
address = '402 E 75th St, New York, NY'
res = geocoder.geocode_google(address)
print res['lat'], res['lng']

address = ['436 E 75th St, New York, New York', '402 E 75th St, New York, NY']
res = [geocoder.geocode_google(address[x]) for x in range(2)]
print res['lat'], res['lng']