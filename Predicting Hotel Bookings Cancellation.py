#!/usr/bin/env python
# coding: utf-8

# # Predicting Hotel Bookings Cancellation

# In[71]:


from   category_encoders          import *
import numpy as np
import pandas as pd
from   sklearn.compose            import *
from   sklearn.ensemble           import *
from   sklearn.experimental       import *
from   sklearn.impute             import *
from   sklearn.linear_model       import *
from   sklearn.metrics            import * 
from   sklearn.pipeline           import *
from   sklearn.preprocessing      import *
from   sklearn.tree               import *
from   sklearn.model_selection    import *
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.base            import BaseEstimator
from imblearn      import pipeline
import imblearn
import eli5
from sklearn.model_selection import RandomizedSearchCV
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import seaborn as sns


# In[3]:


def confusio_matrix(y_test, y_predicted):
    
    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[4]:


hotel = pd.read_csv("hotel_bookings.csv")


# In[5]:


# Dropping irrelevant columns
y=hotel.iloc[:,1]
X=hotel


# In[6]:


# Creating a Train and test dataset. We set a random state so we can have reproducible Train and Test dataset for our project
# We then put the test set away and not peek at it until we chose our model! .
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=10)


# ## Feature engineering and pipeline creation

# In[7]:


# Creating a class dummy estimator for pipeline
class DummyEstimator(BaseEstimator):
    "Pass through class, methods are present but do nothing."
    def fit(self): pass
    def score(self): pass


# In[8]:


# Selecting categorical columns while removing irrelevant columns 
categorical_cols = (X.dtypes == object)
categorical_cols[['is_canceled','company','reservation_status','reservation_status_date','agent']]=False


# Selecting continuous columns while removing irrelevant columns 
continuous_cols = (X.dtypes != object)
continuous_cols[['is_canceled','company','reservation_status','reservation_status_date','agent']]=False

#Applying transformation on columns
#OneHotEncoding Categorical variables
# Scaling the Continuous data
preprocessing = ColumnTransformer([
                                ('categorical', OneHotEncoder(handle_unknown='ignore'),categorical_cols ),
                                ('continuous', QuantileTransformer(), continuous_cols)
                                ])

# Using simpleimputer to handle missing values. We are replacing the values with most frequent values 
pipe = Pipeline([
                ('imputation',SimpleImputer(missing_values=np.nan,strategy="most_frequent")),
                ('preprocessing',preprocessing),
                ('MC',DummyEstimator())]) 


# ## Creating custom evaluation metrics

# In[9]:



from sklearn.metrics import f1_score, make_scorer
f1_weighted= make_scorer(f1_score , average='weighted')
recall_weighted=make_scorer(recall_score,average='weighted')


# ##  Hyperparameter Tuning for Random Forest 

# In[1]:



search_space = [
                {'MC': [RandomForestClassifier()],  # Actual Estimator
                     'MC__criterion': ['gini', 'entropy'],
                     'MC__max_depth':[5,10,15],
                     'MC__min_samples_leaf':[5,10],
                     'MC__class_weight':['balanced'],
                     'MC__n_estimators':[10,50],
                     'MC__bootstrap':[True,False],
                     'MC__max_features':['sqrt','log2']
                     
                     }
                     
                     ]


# ### Cross Validation and Randomized grid search for Random Forest model

# In[ ]:


# Grid Search randomly selects parameters from search_space, and scores the candidate models on validation datasets 
random_search = RandomizedSearchCV(pipe, search_space,scoring={"f1_score":f1_weighted,"recall":recall_weighted},cv=5,refit="f1_score",n_jobs = -1)
search=random_search.fit(X_train,y_train)


# ###  The best estimator found through Randomised Search CV for Random Forest

# In[ ]:


# The best model fitted during cross validation
search.best_estimator_


# In[ ]:


# The best score found during the cross validation
search.best_score_


# ## Hyperparameter Tuning for Logistic Regression

# In[ ]:


##  The following hyperparameters are useful when tuning a Logistic model:
# 1. penalty: Since we have large number of features, regularisation will always help. Either L1 or L2 can give better results
# 2. C: it is the penalty parameter, higher the value, higher the penalty on large coefficients
# 3. class_weight: since we are dealing with an imbalanced dataset, its better to test for 'balanced' parameter
# 4. solver: liblinear solver is generally the fastest. newton-cg uses linear conjugate gradient algo helping 
# to converge to an optimal solution faster, hence we will consider both

search_space_log=[{'MC': [LogisticRegression()], # Actual Estimator
                     'MC__penalty': [ 'l2' ,'l1'],
                     'MC__C': np.logspace(0, 5, 20),
                     'MC__class_weight':['balanced',None],
                     'MC__solver':['liblinear','newton-cg'],
                     'MC__fit_intercept':[True,False]
              }]


# ###  Cross Validation and Randomized grid search for Logistic Regression

# In[ ]:


# Grid Search randomly selects parameters from search_space_log, and scores the candidate models on validation datasets 

random_search = RandomizedSearchCV(pipe, search_space_log,scoring={"f1_score":f1_weighted,"recall":recall_weighted},cv=5,refit="f1_score",n_jobs = -1)
search=random_search.fit(X_train,y_train)


# ### The best estimator found through Randomised Search CV for Logistic Reg

# In[ ]:


# The best model fitted during cross validation
search.best_estimator_


# In[ ]:


# The best score found during the cross validation
search.best_score_


#  ##  Hyperparameter Tuning for  XgBoost
# 

# In[10]:


# according to XgBoost documentation, this should give you an idea about scale_pos_weight parameter used to fit imbalanced datasets
from collections import Counter
Counter(y_train)[0]/(Counter(y_train)[1])


# In[11]:



search_space_xgboost = [
                {'MC': [xgb.XGBClassifier(objective="binary:logistic",use_label_encoder=False)],  # Actual Estimator
                     'MC__max_depth':[10,15],
                     'MC__eta':[0.2,0.3],
                     'MC__scale_pos_weight':[1.65,1.69]
                     }
                     
                     ]


# ###  Cross Validation and Randomized grid search for XgBoost

# In[12]:



random_search = RandomizedSearchCV(pipe, search_space_xgboost,scoring={"f1_score":f1_weighted,"recall":recall_weighted},cv=2,refit="f1_score",n_jobs = -1)
search=random_search.fit(X_train,y_train)


# ### Best estimated XgBoost model

# In[13]:


search.best_estimator_


# In[14]:


search.best_score_


# ### _XgBoost is the winning model!_ We will be moving with this model further on in the analysis 

# 

# ##  Fitting selected model on final train dataset

# In[47]:


pipe=search.best_estimator_


# In[43]:


pipe.fit(X_train,y_train)


# 

# ## Using permutation importance for model interpretation

# In[57]:


# We call permuation_importance and input the fitted model as model parameter
# We call it on test set, since it calculates the feature importance via its performance on our test set
result = permutation_importance(pipe, X_test, y_test, n_repeats=2,
                                random_state=42, n_jobs=-1)


# In[83]:


# We create a dataframe to plot the importance values 
df = pd.DataFrame({'columns':X_test.columns, 'importance':result.importances_mean})
df=df.loc[df['importance']>0].sort_values(['importance'],ascending=False).reset_index()


# In[87]:


# We see country of origin, lead time, and deposit type are the most important variables. this is something we caught on during our EDA as well.
plt.subplots(1,figsize=(13,12))
sns.barplot(x="importance",y="columns",data=df)
plt.show()


# ## Evaluating Model performance on Test set

# In[88]:


y_pred=pipe.predict(X_test)


# In[89]:



print("Weighted f1 score is:",f1_score(y_test, y_pred, average='weighted'))


# In[ ]:



confusio_matrix(y_test, y_pred)


# In[ ]:


# We are able to precisely predict the 0s and 1s with 91% and 85% accuracy.

print(classification_report(y_test, y_pred, target_names=['0','1']))

