# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:11:04 2018

@author: zubov
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:13:06 2018

@author: zubov
"""
########################################################################################################################
#Importing modules
########################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier 

########################################################################################################################
#Getting dataset and data preparation
#######################################################################################################################
 
# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"

# Pulling down the data directly into a pandas dataframe
#Because data in .data file are splitted by commas, read_csv() still works
#Dataset doesn't have a header
census_df = pd.read_csv(url, header=None) 

#Printing first 5 rows 
#function head() shows first 5 rows by default
print (census_df.head())

# Making list of column names
census_columns = ["age",	"class of worker", "industry code", "occupation code", "education", "wage per hour", "enrolled in edu inst last wk", "marital status", "major industry code", "major occupation code", "race", "hispanic origin", "sex", "member of a labor union", "reason for unemployment", "full or part time employment stat", "capital gains",  "capital losses", "dividends from stocks", "tax filer status", "region of previous residence", "state of previous residence", "detailed household and family stat", "detailed household summary in household","instance weight", "migration code-change in msa", "migration code-change in reg", "migration code-move within reg", "live in this house 1 year ago", "migration prev res in sunbelt", "num persons worked for employer", "family members under 18", "country of birth father", "country of birth mother", "country of birth self", "citizenship", "own business or self employed", "fill inc questionnaire for veteran admin", "veterans benefits", "weeks worked in year", "year", "income"]

# Applying list of column names to dataframe
census_df.columns = census_columns

#Printing first 5 rows and checking new column titles
print(census_df.head())

#Preview iinformation about dataset
census_df.info()
census_df.dtypes


#In this dataset missing values could be presented by values " ?" or " Not in universe"
#Change them with nan values

# Replace >Question Marks< with NaNs
census_df = census_df.replace(to_replace=" ?", value=float("NaN"))

# Replace value <Not in universe> with NaNs
census_df = census_df.replace(to_replace=" Not in universe", value=float("NaN"))
census_df = census_df.replace(to_replace=" Not in universe or children", value=float("NaN"))
census_df.info()


#count null values in each column
census_df.isnull().sum()

#Several columns have mostly null values. Probably we could drop them
census_df= census_df.drop(['enrolled in edu inst last wk','member of a labor union', 'reason for unemployment', 'fill inc questionnaire for veteran admin', 'region of previous residence', 'state of previous residence', 'migration prev res in sunbelt', 'family members under 18', 'industry code', 'occupation code', 'detailed household and family stat', 'detailed household summary in household', 'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 'country of birth father', 'country of birth mother', 'veterans benefits', 'instance weight' ], axis=1)
census_df.info()



########################
#In column "country of birth self" there are 3393 null values
#Let's check what country is the most popular

#The most popular country is United States. Replace null values to "United States"
census_df.loc[census_df['country of birth self'].isnull(), 'country of birth self'] = " United States"
census_df.loc[census_df.loc[:, "country of birth self"] != " United-States", "country of birth self"] = " Not US"

####################
#Normalizing dataset
####################

# Normalizing numeric variables using numpy
census_df['age'] =(census_df['age'] - census_df['age'].min())/(census_df['age'].max() - census_df['age'].min())
census_df['wage per hour'] =(census_df['wage per hour'] - census_df['wage per hour'].min())/(census_df['wage per hour'].max() - census_df['wage per hour'].min())
census_df['capital gains'] =(census_df['capital gains'] - census_df['capital gains'].min())/(census_df['capital gains'].max() - census_df['capital gains'].min())
census_df['capital losses'] =(census_df['capital losses'] - census_df['capital losses'].min())/(census_df['capital losses'].max() - census_df['capital losses'].min())
census_df['weeks worked in year'] =(census_df['weeks worked in year'] - census_df['weeks worked in year'].min())/(census_df['weeks worked in year'].max() - census_df['weeks worked in year'].min())
census_df['num persons worked for employer'] =(census_df['num persons worked for employer'] - census_df['num persons worked for employer'].min())/(census_df['num persons worked for employer'].max() - census_df['num persons worked for employer'].min())
census_df['dividends from stocks'] =(census_df['dividends from stocks'] - census_df['dividends from stocks'].min())/(census_df['dividends from stocks'].max() - census_df['dividends from stocks'].min())


#Binning categorical variable "education" 

#Split variable "education" for several bins:

#"Children" - category "Children"
#"Elementary school" - categories "Less than 1st grade", "1st 2nd 3rd or 4th grade"
#"Middle school" - categories "5th or 6th grade", "7th and 8th grade"
#"High school" - categories "9th grade", "10th grade", "11th grade", "12th grade no diploma", "High school graduate"
#"Associates degree" - categories "Associates degree-occup /vocational", "Associates degree-academic program"
#"Prof school degree (MD DDS DVM LLB JD)" - categories "Professional school degree"
#"Some college but no degree" - categories "Some college but no degree"
#"Bachelors degree" - category "Bachelors degree(BA AB BS)"
#"Masters degree" - categories "Masters degree(MA MS MEng MEd MSW MBA)"
#"Doctorate degree" - categories "Doctorate degree(PhD EdD)"

census_df.loc[census_df.loc[:, "education"] == " 9th grade", "education"] = "High school"
census_df.loc[census_df.loc[:, "education"] == " 10th grade", "education"] = "High school"
census_df.loc[census_df.loc[:, "education"] == " 11th grade", "education"] = "High school"
census_df.loc[census_df.loc[:, "education"] == " 12th grade no diploma", "education"] = "High school"
census_df.loc[census_df.loc[:, "education"] == " High school graduate", "education"] = "High school"

census_df.loc[census_df.loc[:, "education"] == " 5th or 6th grade", "education"] = "Middle school" 
census_df.loc[census_df.loc[:, "education"] == " Less than 1st grade", "education"] = "Elementary school"
census_df.loc[census_df.loc[:, "education"] == " 1st 2nd 3rd or 4th grade", "education"] = "Elementary school"

census_df.loc[census_df.loc[:, "education"] == " 7th and 8th grade", "education"] = "Middle school"

census_df.loc[census_df.loc[:, "education"] == " Associates degree-occup /vocational", "education"] = "Associates degree"
census_df.loc[census_df.loc[:, "education"] == " Associates degree-academic program", "education"] = "Associates degree"

census_df.loc[census_df.loc[:, "education"] == " Some college but no degree", "education"] = "College, no degree"

census_df.loc[census_df.loc[:, "education"] == " Prof school degree (MD DDS DVM LLB JD)", "education"] = "Professional school degree"

census_df.loc[census_df.loc[:, "education"] == " Bachelors degree(BA AB BS)","education"] = "Bachelors degree"
census_df.loc[census_df.loc[:, "education"] == " Masters degree(MA MS MEng MEd MSW MBA)", "education"] = "Masters degree"
census_df.loc[census_df.loc[:, "education"] == " Doctorate degree(PhD EdD)", "education"] = "Doctorate degree"
# Plot the counts for each category
census_df.loc[:,"education"].value_counts().plot(kind='bar')


#Creating dummy variables

# Create new columns, one for each state in columns "education", "marital status", "class of worker", "major industry code", "major occupation code", "race", "hispanic origin", "sex", "full or part time employment stat", "tax filer status", "year", "country of birth self", "citizenship"
edu_dummies = pd.get_dummies(census_df['education'], prefix = 'education')
census_df = pd.concat([census_df, edu_dummies], axis=1)

marst_dummies= pd.get_dummies(census_df['marital status'], prefix = 'marital status', dummy_na=False)
census_df = pd.concat([census_df, marst_dummies], axis=1)

class_dummies = pd.get_dummies(census_df['class of worker'], prefix = 'class of worker', dummy_na=False)
census_df = pd.concat([census_df, class_dummies], axis=1)

industry_dummies = pd.get_dummies(census_df['major industry code'], prefix = 'major industry code', dummy_na=False)
census_df = pd.concat([census_df, industry_dummies], axis=1)

occupation_dummies = pd.get_dummies(census_df['major occupation code'], prefix = 'major occupation code', dummy_na=False)
census_df = pd.concat([census_df, occupation_dummies], axis=1)

race_dummies = pd.get_dummies(census_df['race'], prefix = 'race', dummy_na=False)
census_df = pd.concat([census_df, race_dummies], axis=1)

hispanic_dummies = pd.get_dummies(census_df['hispanic origin'], prefix = 'hispanic origin', dummy_na=False)
census_df = pd.concat([census_df, hispanic_dummies], axis=1)

sex_dummies = pd.get_dummies(census_df['sex'], prefix = 'sex', dummy_na=False)
census_df = pd.concat([census_df, sex_dummies], axis=1)

emp_dummies = pd.get_dummies(census_df['full or part time employment stat'], prefix = 'full or part time employment stat', dummy_na=False)
census_df = pd.concat([census_df, emp_dummies], axis=1)

tax_dummies = pd.get_dummies(census_df['tax filer status'], prefix = 'tax filer status',dummy_na=False)
census_df = pd.concat([census_df, tax_dummies], axis=1)

year_dummies = pd.get_dummies(census_df['year'], prefix = 'year', dummy_na=False)
census_df = pd.concat([census_df, year_dummies], axis=1)

country_dummies = pd.get_dummies(census_df['country of birth self'], prefix = 'country of birth self', dummy_na=False)
census_df = pd.concat([census_df, country_dummies], axis=1)

citizenship_dummies = pd.get_dummies(census_df['citizenship'], prefix = 'citizenship', dummy_na=False)
census_df = pd.concat([census_df, citizenship_dummies], axis=1)


#Income - is a target variable. Let's create new column and decode "income" variable (1 - if income > $50000, 0 - if income < $50000)

census_df.loc[:, "income > $50000"] = (census_df.loc[:, "income"] == " 50000+.").astype(int)

#Removing obsolete columns

# Remove obsolete columns "education", "income","marital status", "class of worker", "major industry code", "major occupation code", "race", "hispanic origin", "sex", "full or part time employment stat", "tax filer status", "year", "country of birth self", "citizenship"
#For all those variables were created dummy variables (exept income, which was just decoded), so we don't need them anymore
census_df = census_df.drop("income", axis=1)
census_df = census_df.drop("education", axis=1)
census_df = census_df.drop("marital status", axis=1)
census_df = census_df.drop("class of worker", axis=1)
census_df = census_df.drop("major industry code", axis=1)
census_df = census_df.drop("major occupation code", axis=1)
census_df = census_df.drop("race", axis=1)
census_df = census_df.drop("hispanic origin", axis=1)
census_df = census_df.drop("sex", axis=1)
census_df = census_df.drop("full or part time employment stat", axis=1)
census_df = census_df.drop("tax filer status", axis=1)
census_df = census_df.drop("year", axis=1)
census_df = census_df.drop("country of birth self", axis=1)
census_df = census_df.drop("citizenship", axis=1)



##############################
#Getting features and targets
##############################
features = census_df.iloc[:, :-1]
target = census_df.iloc[:, -1].values

#from sklearn import preprocessing
#features = preprocessing.normalize(features)

#######################################################################################################################
#Splitting dataset onto train set and test set
######################################################################################################################


X, XX, Y, YY = train_test_split(features, target, test_size = 0.2, random_state = 0)


#Feature selection

#There are too many features in dataset, what affects speed of algorithm
#Select 10 significant features

from sklearn.feature_selection import SelectKBest, f_classif
x_data_kbest = SelectKBest(f_classif, k=10).fit(X, Y) #10 significant variables for classification model

result = pd.DataFrame()
#Get full list of features
features_list = list(features.columns.values)
#Get significant features
result['columns'] = features_list
result['value'] = x_data_kbest.get_support()

result.to_csv('nbc_result.csv', sep=",")

#According the results leave in train and test sets only chosen significant features

X = X[['capital gains','num persons worked for employer','weeks worked in year','education_Bachelors degree','education_Masters degree','education_Professional school degree','major occupation code_ Executive admin and managerial','major occupation code_ Professional specialty','tax filer status_ Joint both under 65','tax filer status_ Nonfiler']]
XX = XX[['capital gains','num persons worked for employer','weeks worked in year','education_Bachelors degree','education_Masters degree','education_Professional school degree','major occupation code_ Executive admin and managerial','major occupation code_ Professional specialty','tax filer status_ Joint both under 65','tax filer status_ Nonfiler']]
#X.shape
#######################################################################################################################
#In this assignment will be used Naive Bayes classifier, Desicion Tree classifier and Random Forest classifier
#######################################################################################################################

######################################################################################################################
# Naive Bayes classifier
######################################################################################################################

print ('\n\nNaive Bayes classifier\n')
clf_bayes = GaussianNB() # with default parameters
#Train decision tree classification model
clf_bayes.fit(X, Y)

#Apply model to test set
print ("predictions for test set:")
y_pred_class_bayes = clf_bayes.predict(XX)
print(y_pred_class_bayes)
print ('actual class values:')
print (YY)


#Confusion matrix

print(confusion_matrix(YY, y_pred_class_bayes))

# Accuracy
print(accuracy_score(YY, y_pred_class_bayes)) 


# Error rate
print(1 - accuracy_score(YY, y_pred_class_bayes)) 

# Recall 
print(recall_score(YY, y_pred_class_bayes))
# Precision
print(precision_score(YY, y_pred_class_bayes)) 

#f1_score
print(f1_score(YY, y_pred_class_bayes)) 

#Alternative function for getting metrics
report_bayes = classification_report(YY, y_pred_class_bayes)
print(report_bayes)


#AUC analysis and probability threshholds for Naive Bayes classification model

# store the predicted probabilities for class 1
#Class 1 means that person earn > $50000 per year
y_pred_prob_bayes = clf_bayes.predict_proba(XX)[:, 1]


fpr_bayes, tpr_bayes, thresholds_bayes = roc_curve(YY, y_pred_prob_bayes)


print ("\nTP rates:", np.round(tpr_bayes, 2))
print ("\nFP rates:", np.round(fpr_bayes, 2))
print ("\nProbability thresholds:", np.round(thresholds_bayes, 2))

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
AUC_bayes = auc(fpr_bayes, tpr_bayes)
print(AUC_bayes)

#Plot AUC
plt.plot(fpr_bayes, tpr_bayes)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for census Naive Bayes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


######################################################################################################################
# Decision Tree classifier
######################################################################################################################

print ('\n\nDecision Tree classifier\n')
clf_tree = DecisionTreeClassifier() # with default parameters
#Train decision tree classification model
clf_tree.fit(X, Y)

#Apply model to test set
print ("predictions for test set:")
y_pred_class_tree = clf_tree.predict(XX)
print(y_pred_class_tree)
print ('actual class values:')
print (YY)


#Confusion matrix

print(confusion_matrix(YY, y_pred_class_tree))

CM_log = confusion_matrix(YY, y_pred_class_tree)

tn, fp, fn, tp = CM_log.ravel()

# Accuracy
print(accuracy_score(YY, y_pred_class_tree)) #good result


# Error rate
print(1 - accuracy_score(YY, y_pred_class_tree)) #good result

# Recall 
print(recall_score(YY, y_pred_class_tree)) #this metrics looks bad

# Precision
print(precision_score(YY, y_pred_class_tree)) #pretty good result


#f1_score
print(f1_score(YY, y_pred_class_tree)) 

#Alternative function for getting metrics
report_tree = classification_report(YY, y_pred_class_tree)
print(report_tree)


#AUC analysis and probability threshholds for decision tree classification model

# store the predicted probabilities for class 1
#Class 1 means that person earn > $50000 per year
y_pred_prob_tree = clf_tree.predict_proba(XX)[:, 1]


fpr_tree, tpr_tree, thresholds_tree = roc_curve(YY, y_pred_prob_tree)


print ("\nTP rates:", np.round(tpr_tree, 2))
print ("\nFP rates:", np.round(fpr_tree, 2))
print ("\nProbability thresholds:", np.round(thresholds_tree, 2))

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
AUC_tree = auc(fpr_tree, tpr_tree)
print(AUC_tree)

#Plot AUC
plt.plot(fpr_tree, tpr_tree)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for census decision classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

######################################################################################################################
# Random Forest classifier
######################################################################################################################

estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf_forest = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf_forest.fit(X, Y)


print ("predictions for test set:")
y_pred_class_forest = clf_forest.predict(XX)
print(y_pred_class_forest)
print ('actual class values:')
print (YY)


#Confusion matrix

print(confusion_matrix(YY, y_pred_class_forest))

CM_log = confusion_matrix(YY, y_pred_class_forest)

tn, fp, fn, tp = CM_log.ravel()

# Accuracy
print(accuracy_score(YY, y_pred_class_forest))

# Error rate
print(1 - accuracy_score(YY, y_pred_class_forest))

# Recall 
print(recall_score(YY, y_pred_class_forest)) #this metrics looks bad but better then for decision tree classifier

# Precision
print(precision_score(YY, y_pred_class_forest))


#f1_score
print(f1_score(YY, y_pred_class_forest))

#Alternative function for getting metrics
report_forest = classification_report(YY, y_pred_class_forest)
print(report_forest)


#AUC analysis and probability threshholds for random forest classification model


# store the predicted probabilities for class 1
#Class 1 means that person earn > $50000 per year
y_pred_prob_forest = clf_forest.predict_proba(XX)[:, 1]


fpr_forest, tpr_forest, thresholds_forest = roc_curve(YY, y_pred_prob_forest)


print ("\nTP rates:", np.round(tpr_forest, 2))
print ("\nFP rates:", np.round(fpr_forest, 2))
print ("\nProbability thresholds:", np.round(thresholds_forest, 2))

#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
AUC_forest = auc(fpr_forest, tpr_forest)
print(AUC_forest)

#Plot AUC
plt.plot(fpr_forest, tpr_forest)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for census random forest classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)



#############################################
#SUMMARY
#############################################

#Chosen models (Desicion Tree Classifier and Random Forest classifier) shows good results in accuracy and precision what means that mostly predicted positive values were really positive,
#but bad recall,  so model didn't find all positive values.

print('\n Naive Bayes\n')
print(report_bayes)

print('\n Decision Tree metrices\n')
print(report_tree)

print('\n Random Forest metrices\n')
print(report_forest)

#Naive Bayes shows better results in recall (for class 1) and AUC-score, but precision is more weak
#Both other models (decision tree and random forest) shows similar results, but scores for random forest model are a bit higher, so I would choose it 
#Also if we suppose that target value is "person earns less than $50000 per year", models would work better
#Better results could be got by using different amount of significant features and classificators parameters 
#(unfortunately, I didn't have enough time for that)  
#Final choice of classifier should depend on our goal, what exactly we want to get from the data