#!/usr/bin/env python
# coding: utf-8

# In[563]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import pydot

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[599]:


data1 = pd.read_csv('C:\\Users\\Dhiru\\Desktop\\data science\\data1.csv')


# In[600]:


data1.head(5)


# In[601]:


data1.describe()


# In[739]:


data1.describe().to_csv("C:\\Users\\Dhiru\\Desktop\\data science\\my_description.csv")


# In[740]:


data1 = pd.DataFrame(data1)


# In[741]:


df = data1


# In[742]:


data1.dtypes


# In[745]:


data1.dtypes


# In[747]:


data1.isnull().sum()


# In[748]:


sns.heatmap(data1.isnull(),cmap='viridis')


# ## convert gender in to three catagory i.e male, female and non

# In[749]:


for i in range(0,10000):
    if(type(data1.iloc[i,1]) == float):
        data1.iloc[i,1] = "G"


# In[750]:


data1.iloc[4,1] == 'f'


# In[753]:


plt.figure(figsize=(15,8))
ax1=sns.countplot(x = 'app_downloads',hue='gender',data=data1)
legend_labels, _= ax1.get_legend_handles_labels()
ax1.legend(legend_labels, ['no gender','man','woman'], bbox_to_anchor=(1,1))


# In[754]:


plt.figure(figsize=(15,8))
ax2=sns.countplot(x = 'unique_offer_clicked',hue='gender',data=data1)
legend_labels, _= ax2.get_legend_handles_labels()
ax2.legend(legend_labels, ['no gender','man','woman'], bbox_to_anchor=(1,1))


# In[755]:


plt.figure(figsize=(15,8))
ax2=sns.countplot(x = 'unique_offer_impressions',hue='gender',data=data1)
legend_labels, _= ax2.get_legend_handles_labels()
ax2.legend(legend_labels, ['no gender','man','woman'], bbox_to_anchor=(1,1))


# In[756]:


from pylab import *


# In[757]:


plt.figure(figsize=(20,10))

subplot(4,4,1)
title('subplot(2,2,1)')
sns.boxplot(x='gender',y='app_downloads',data=data1)

subplot(4,4,2)
title('subplot(2,2,2)')
sns.boxplot(x='gender',y='unique_offer_clicked',data=data1)

subplot(4,4,3)
title('subplot(2,2,3)')
sns.boxplot(x='gender',y='total_offer_clicks',data=data1)

subplot(4,4,4)
title('subplot(2,2,4)')
sns.boxplot(x='gender',y='unique_offer_impressions',data=data1)

subplot(4,4,5)
title('subplot(2,2,5)')
sns.boxplot(x='gender',y='total_offer_impressions',data=data1)

subplot(4,4,6)
title('subplot(2,2,6)')
sns.boxplot(x='gender',y='total_offers_redeemed',data=data1)


# In[764]:


plt.plot(data1['unique_offer_clicked'],data1['total_offer_clicks'],'o',color='red')


# In[765]:


plt.plot(data1['unique_offer_impressions'],data1['total_offer_impressions'],'o',color='red');


# In[766]:


plt.plot(data1['min_redemptions'],data1['total_offers_redeemed'],'o',color='red');


# In[767]:


plt.plot(data1['max_redemptions'],data1['total_offers_redeemed'],'o',color='red');


# In[768]:


list = ['consumer_id','has_gender','has_first_name','has_last_name','has_email','has_dob','customer_age','account_status']
data2 = data1.drop(list,axis=1)
data2.shape


# In[769]:


plt.figure(figsize=(20,20))
fig, ax = plt.subplots()
corr = data2.corr(method='pearson')
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
#plt.show()
#plt.savefig('corrplot.png')
#yticklabels=corr.columns.values,
sns.heatmap(corr, xticklabels=corr.columns.values,annot=True, fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# ### unique_click_offer is strongly correlated with total_click_offer

# In[770]:


from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


# In[771]:


#Passing the values of the dataset to Min-Max-Scaler
X=data2.drop(['gender','unique_offer_clicked','unique_offer_impressions','min_redemptions','max_redemptions'],axis=1)
data_values = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data_values)


# In[772]:


x_scaled


# In[773]:


# k means determine k

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x_scaled)
    kmeanModel.fit(x_scaled)
    distortions.append(sum(np.min(cdist(x_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x_scaled.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[774]:



kmeans = KMeans(n_clusters=2) # You want cluster the records into 2: fraud or genuine
kmeans.fit(x_scaled)


# In[775]:


y_means = kmeans.predict(x_scaled)


# In[776]:


y_means


# In[777]:


from sklearn.metrics import silhouette_score


# In[778]:


print(f'Silhouette Score(n=2): {silhouette_score(x_scaled, y_means)}')


# In[779]:


X['predicted_values'] = y_means


# In[780]:


X.predicted_values.value_counts()


# In[781]:


#plt.scatter(dataset, X.loc[:, 'total_offers_redeemed'], c=y_means, s=50, cmap='viridis')

#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[782]:


X[X.predicted_values == 1].describe()


# In[783]:


X[X.predicted_values == 1].describe().to_csv("C:\\Users\\Dhiru\\Desktop\\data science\\predicted_description.csv")


# In[784]:


df['Predicted'] = y_means


# ## Accounts which are fraud

# In[785]:


df[df.Predicted == 1]


# In[786]:


pca_reducer = PCA(n_components =1)
dataset = pca_reducer.fit_transform(X)


# In[787]:


# k means determine k

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(dataset)
    kmeanModel.fit(dataset)
    distortions.append(sum(np.min(cdist(dataset, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dataset.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[788]:


kmeans = KMeans(n_clusters=2) # You want cluster the records into 2: fraud or genuine
kmeans.fit(dataset)


# In[789]:


y_means = kmeans.predict(dataset)


# In[790]:


print(f'Silhouette Score(n=2): {silhouette_score(dataset, y_means)}')


# In[791]:



X['predicted_values'] = y_means


# In[792]:


X.predicted_values.value_counts()


# In[793]:


X[X.predicted_values == 1].describe()


# In[794]:


X[X.predicted_values == 1].describe().to_csv("C:\\Users\\Dhiru\\Desktop\\data science\\predicted_description2.csv")


# In[ ]:


df['Predicted'] = y_means


# In[ ]:


df[df.Predicted == 1]


# In[795]:


##### fig, ax = plt.subplots()


plt.scatter(dataset, X.loc[:, 'total_offers_redeemed'], c=y_means, s=50, cmap='viridis')

centers = kmeans.cluster_centers_

#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[796]:


data1['Predicted'] = y_means


# In[797]:


data1[data1.Predicted == 1]


# In[798]:


data1[data1.Predicted == 1].describe()


# ## Random Forest

# In[799]:


# Import train_test_split function
from sklearn.model_selection import train_test_split


# In[800]:


#data2 = pd.read_csv('C:\\Users\\Dhiru\\Desktop\\data science\\data1.csv')


# In[852]:


data2


# In[853]:


#Gender = data2.gender


# In[854]:


data2.drop('gender',axis=1,inplace=True)


# In[804]:


#data2


# In[805]:


#min_max_scaler = preprocessing.MinMaxScaler()
#data2_scaled = min_max_scaler.fit_transform(data2)


# In[855]:


y = data2['total_offers_redeemed']
X = data2.drop(['total_offers_redeemed'], axis=1)


# In[857]:


y1_1=y
X1_1=X


# In[858]:


X.dtypes


# In[859]:


#y = data2['total_offers_redeemed']
#X = data2.drop(['total_offers_redeemed','unique_offer_impressions','unique_offer_clicked','min_redemptions','max_redemptions'], axis=1)


# In[860]:


X.columns


# In[861]:


X_list = X.columns.tolist()


# In[862]:


X_list


# In[863]:


X1 = X


# In[864]:


y = np.array(y)
X = np.array(X)


# In[865]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[866]:


print('Training X_train Shape:', X_train.shape)
print('Training y_train Shape:', y_train.shape)
print('Testing X_test Shape:', X_test.shape)
print('Testing y_test Shape:', y_test.shape)


# In[867]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


# In[868]:


#from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import make_classification


# In[869]:


# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000,
#                           n_features=10,
#                           n_informative=3,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=0,
#                           shuffle=False)


# In[870]:


#param_grid = {'n_estimators': [200, 700],
#    'max_features': ['auto', 'sqrt', 'log2']}


# In[871]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, random_state=123,class_weight="balanced")


# In[872]:


#clf.fit(X_train,y_train)


# In[873]:


#CV_clf = GridSearchCV(clf,param_grid=param_grid,cv=5)


# In[874]:


#CV_clf.fit(X,y)


# In[875]:


#print CV_clf.best_params_


# In[876]:


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[877]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[878]:


errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'total_offers_redeemed')


# In[881]:


mape = np.mean(100 * (errors / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
mape


# In[882]:


from sklearn.tree import export_graphviz
import pydot


# In[883]:


# Pull out one tree from the forest
tree = clf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[884]:


# Limit depth of tree to 3 levels
rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)
rf_small.fit(X_train,y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = X_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');


# In[885]:


clf.feature_importances_


# In[886]:


# Get numerical feature importances
importances = clf.feature_importances_.tolist()
# List of tuples with variable and importance
feature_importances = [(X1, round(importance, 2)) for X1, importance in zip(X_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[887]:


[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[888]:


[*range(len(importances))]


# In[889]:


# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = [*range(len(importances))]
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, X_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[890]:


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification


# ### Random Forest CV

# In[839]:


data2


# In[ ]:





# ### Logistic Regression

# In[891]:



from sklearn.linear_model import LogisticRegression


# In[892]:


logreg = LogisticRegression(solver='liblinear', max_iter=200)


# In[842]:


#data1 = pd.read_csv('C:\\Users\\Dhiru\\Desktop\\data science\\data1.csv')
#data1 = pd.DataFrame(data1)


# In[843]:


#y = data1['total_offers_redeemed']
#X = data1.drop(['total_offers_redeemed','consumer_id','has_gender','has_first_name','has_first_name','has_last_name','has_email','has_dob','customer_age','avg_redemptions'], axis=1)


# In[893]:


y = y1_1
X = X1_1


# In[894]:


#from sklearn import preprocessing
#y = preprocessing.label_binarize(y, classes=[0, 1, 2, 3])
#from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# In[895]:


#y.shape


# In[896]:


# Binarize the output
#y = label_binarize(y, classes=[0, 1, 2])
#n_classes = 10000


# In[897]:


y = np.array(y)
X = np.array(X)


# In[898]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


# In[901]:


result=logreg.fit(X_train,y_train)


# In[902]:


y_pred=logreg.predict(X_test)


# In[903]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[904]:


logreg.intercept_


# In[905]:


logreg.coef_


# In[910]:


logreg.score(X,y)


# In[911]:


from sklearn.metrics import classification_report


# In[912]:


report = classification_report(y_test, y_pred)


# In[913]:


print('report:', report, sep='\n')


# In[ ]:




