# %% [markdown]
# ## Loading the dataset

# %%
# !pip install --upgrade gdown

# %%
# !gdown --id 12vfq3DYFId3bsXuNj_PhsACMzrLTfObs

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

# %%
data=pd.read_csv("data_regression.csv")
# get the first 10 rows
data.head(10)

# %%
# check for the missing values and dataframes
def datainspection(dataframe):
  print("Types of the variables we are working with:")
  print(dataframe.dtypes)
  
  print("Total Samples with missing values:")

  print(data.isnull().any(axis=1).sum()) # null values

  print("Total Missing Values per Variable")
  print(data.isnull().sum())
  print("Map of missing values")
  sns.heatmap(dataframe.isnull())

# %%
datainspection(data)

# %%
data = data.dropna() # cleaning up null values

# %%
# function for encoding categorical variables
def encode_cat(data, vars):
  ord_en = OrdinalEncoder() 
  for v in vars:
    name = v+'_code' # add _code for encoded variables
    data[name] = ord_en.fit_transform(data[[v]])
    print('The encoded values for '+ v + ' are:')
    print(data[name].unique())
  return data
data.head()

# %%
# check for the encoded variables
data = encode_cat(data, ['gender', 'multi_screen', 'mail_subscribed'])
data.head()

# %%
def full_plot(data, class_col, cols_to_exclude):
  cols = data.select_dtypes(include=np.number).columns.tolist() # finding all the numerical columns from the dataframe
  X = data[cols] # creating a dataframe only with the numerical columns
  X = X[X.columns.difference(cols_to_exclude)] # columns to exclude
  X = X[X.columns.difference([class_col])]
  sns.pairplot(data, hue=class_col)

# %%
full_plot(data,class_col='churn', cols_to_exclude=['customer_id','phone_no', 'year'])

# %%
# function for creating plots for selective columns only
def selected_diagnotic(data,class_col, cols_to_eval):
  cols_to_eval.append(class_col) 
  X = data[cols_to_eval] # only selective columns
  sns.pairplot(X, hue=class_col) # plot

# %%
selected_diagnotic(data, class_col='churn', cols_to_eval=['videos_watched', 'no_of_days_subscribed'])

# %%
def logistic_regression(data, class_col, cols_to_exclude):
  cols = data.select_dtypes(include=np.number).columns.tolist() 
  X = data[cols]
  X = X[X.columns.difference([class_col])] 
  X = X[X.columns.difference(cols_to_exclude)] # unwanted columns 

  y = data[class_col] # the target variable 
  logit_model = sm.Logit(y,X) 
  result = logit_model.fit() # fit the model 
  print(result.summary2()) # check for summary 

# %%
logistic_regression(data, class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])

# %%
def prepare_data(data, class_col, cols_to_exclude):
  ## Split in training and test set
  ## Selecting only the numerical columns and excluding the columns we specified in the function
  cols = data.select_dtypes(include=np.number).columns.tolist() 
  X = data[cols]
  X = X[X.columns.difference([class_col])] 
  X = X[X.columns.difference(cols_to_exclude)]
  ## Selecting y as a column
  y = data[class_col]
  return train_test_split(X, y, test_size=0.3, random_state=0) # perform train test split

# %%
def run_model(X_train, X_test, y_train, y_test):
  # Fitting the logistic regression
  logreg = LogisticRegression(random_state=13)
  logreg.fit(X_train, y_train) # fit the model
  # Predicting y values
  y_pred = logreg.predict(X_test) # make predictions on th test data
  logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
  print(classification_report(y_test, y_pred)) # check for classification report 
  print("The area under the curve is:", logit_roc_auc)  # check for AUC
  return y_pred

# %%
X_train, X_test, y_train, y_test = prepare_data(data, class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])
y_pred = run_model(X_train, X_test, y_train, y_test)

# %%
from sklearn.metrics import confusion_matrix

def confusion_m(y_test, y_pred):
  cm = confusion_matrix(y_test, y_pred)
  print(cm)
  tn, fp, fn, tp = cm.ravel()
  print("TN:", tn)
  print("TP:", tp)
  print("FN:", fn)
  print("FP:", fp)

# %%
## Call the function
confusion_m(y_test, y_pred)

# %%
# class imbalance method 1 
def run_model_bweights(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(random_state=13, class_weight='balanced') # define class_weight parameter
    logreg.fit(X_train, y_train) # fit the model 
    y_pred = logreg.predict(X_test) # predict on test data
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test)) # ROC AUC score
    print(classification_report(y_test, y_pred)) 
    print("The area under the curve is:", logit_roc_auc) # AUC curve

# %%
run_model_bweights(X_train, X_test, y_train, y_test)

# %%
# class imbalance method 2
def run_model_aweights(X_train, X_test, y_train, y_test, w):
    logreg = LogisticRegression(random_state=13, class_weight=w) # define class_weight parameter
    logreg.fit(X_train, y_train) # fit the model 
    y_pred = logreg.predict(X_test) # predict on test data
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))  # ROC AUC score
    print(classification_report(y_test, y_pred))
    print("The area under the curve is: %0.2f"%logit_roc_auc)  # AUC curve

# %%
run_model_aweights(X_train,X_test,y_train,y_test,{0:90, 1:10})

# %%
# class imbalance method 3
def adjust_imbalance(X_train, y_train, class_col):
  X = pd.concat([X_train, y_train], axis=1)
  # separate the 2 classes. Here we divide majority and minority classes
  class0 = X[X[class_col] == 0]
  class1 = X[X[class_col] == 1]
  # Case 1 - bootstraps from the minority class
  if len(class1)<len(class0):
    resampled = resample(class1,
                              replace=True, # Upsampling with replacement
                              n_samples=len(class0), ## Number to match majority class
                              random_state=10) 
    resampled_data = pd.concat([resampled, class0]) ## # Combination of majority and upsampled minority class
  # Case 1 - resamples from the majority class
  else:
    resampled = resample(class1,
                              replace=False, ## false instead of True like above
                              n_samples=len(class0), 
                              random_state=10) 
    resampled_data = pd.concat([resampled, class0])
  return resampled_data

# %%
## Call the function
resampled_data = adjust_imbalance(X_train, y_train, class_col='churn')

# %%
X_train, X_test, y_train, y_test = prepare_data(resampled_data, class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])
run_model(X_train, X_test, y_train, y_test)

# %%
def prepare_data_smote(data,class_col,cols_to_exclude):
  # Synthetic Minority Oversampling Technique. 
  # Generates new instances from existing minority cases that you supply as input. 
  cols = data.select_dtypes(include=np.number).columns.tolist() 
  X = data[cols]
  X = X[X.columns.difference([class_col])]
  X = X[X.columns.difference(cols_to_exclude)]
  y = data[class_col]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
  sm = SMOTE(random_state=0, sampling_strategy=1.0)
  # run SMOTE on training set only
  X_train, y_train = sm.fit_resample(X_train, y_train)
  return X_train, X_test, y_train, y_test

# %%
X_train, X_test, y_train, y_test = prepare_data_smote(data,class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])
run_model(X_train, X_test, y_train, y_test)


