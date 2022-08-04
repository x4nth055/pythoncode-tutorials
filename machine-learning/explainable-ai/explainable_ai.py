# %% [markdown]
# ## Explaining Logistic Regression Moel with SHAP
# Logistic regression is often used to predict the probability of binary or multinomial outcomes. it can be multinomial(where more than two outcomes are also possible).
# Class separation is complex in a multinomial class classification model.
# A logistic regression model assumes a logarithmic relationship between the dependent and independent variables, while a linear regression model assumes a linear relationship. 
# The variable of interest in many real-life settings is categorical: The purchase or non-purchase of a product, the approval or non-approval of a credit card, or the cancerousness of a tumor.
# Logistic regression can estimate the likelihood of a case belonging to a specific level in the dependant variable. 
# The logistic regression model can be explained using the following equation:
# {Formula]
# 
# The outcome's log-odds are given by the formula Ln (P/1-P).
# According to the preceding equation's beta coefficients, the outcome variable's probabilities increase or decrease by one unit when the explanatory variable rises or falls. 
# The interpretation of a logistic regression model differs significantly from the interpretation of a linear regression model.
# The right-hand side equation's weighted sum is turned into a probability value. The log-odds are used to describe the value on the left side of the equation. 
# It is termed the log odds because it represents the ratio of an event occurring to the probability of an event not occurring.
# To comprehend the logistic regression model and how decisions are made, it is necessary to grasp the concepts of probabilities and odds.
# You'll utilize churndata.csv, a file in the telecommunications category with 3,333 entries and 18 distinct features. 
# 
# 
#  
# 
#  
#        

# %%
!pip install shap
!pip install LIME
!pip install interpret-core


# %%
#mount drive
%cd ..
from google.colab import drive
drive.mount('/content/gdrive')

# Execute the following command to build a symbolic link, so that the path /content/gdrive/My Drive/ is now equivalent to /mydrive.
!ln -s /content/gdrive/My\ Drive/ /mydrive

# list the contents of /mydrive
!ls /mydrive

#Navigate to /mydrive/churn
%cd /mydrive/churn/

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import interpret.glassbox
import xgboost
import shap
import lime
import lime.lime_tabular
import sklearn
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# As a first stage, you get the data, then convert specific features already in string format using a label encoder.
# You divide the data into 80 percent for training and 20 percent for testing following the transformation.
# To keep the classes balanced, maintain the percentage of churn and no-churn cases while generating the train/test split.
# The model is then trained, and the learned model is applied to the test data.

# %%
data = pd.read_csv('/mydrive/churn/Telecom_Train.csv')
data.head()

# %%
del data['Unnamed: 0'] ## delete Unnamed: 0
le = LabelEncoder() ## perform label encoding
data['area_code_tr'] = le.fit_transform(data['area_code'])
del data['area_code'] ## delete area_code
data['churn_dum'] = pd.get_dummies(data.
churn,prefix='churn',drop_first=True)
del data['international_plan'] ## delete international_plan
del data['voice_mail_plan'] ## delete voice_mail_plan
del data['churn'] ## delete churn
data.info()
data.columns

# %%
X = data[['account_length', 'number_vmail_messages', 'total_day_minutes',
'total_day_calls', 'total_day_charge', 'total_eve_minutes',
'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
'total_night_calls', 'total_night_charge', 'total_intl_minutes',
'total_intl_calls', 'total_intl_charge',
'number_customer_service_calls', 'area_code_tr']]
Y = data['churn_dum']
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.20,stratify=Y)
l_model = LogisticRegression(max_iter=10000)
l_model.fit(xtrain,ytrain)
print("training accuracy:", l_model.score(xtrain,ytrain)) #training accuracy
print("test accuracy:",l_model.score(xtest,ytest)) # test accuracy


# %%
print(np.round(l_model.coef_,2))## Coeffiscient 
print(l_model.intercept_) ## intercept

# %% [markdown]
# 
# Only the area code is transformed. The remaining features are either integers or floating-point numbers to train the model.
# It's now possible to see how a prediction is made by examining the distribution of probabilities, the log odds, the odds ratios, and other model parameters.
# SHAP values may reveal strong interaction effects when used to explain the probability of a logistic regression model.
# 

# %% [markdown]
# The appropriate output may be generated using two new utility functions that you built and which can then be used in a visual representation of SHAP values. 

# %%
# Provide Probability as Output
def m_churn_proba(x):
   return l_model.predict_proba(x)[:,1]
# Provide Log Odds as Output
def model_churn_log_odds(x):
   p = l_model.predict_log_proba(x)
   return p[:,1] - p[:,0]

# %% [markdown]
# The partial dependency plot for the feature total day in minutes for record number 25 demonstrates a positive but not linear relationship between the function's probability value or predicted value and the feature.

# %%
# make a standard partial dependence plot
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"total_day_minutes", m_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False)

# %% [markdown]
# Any machine learning model or Python function may be explained using Shapley values. This is the SHAP library's primary explainer interface.
# It accepts any model and masker combination and produces a callable subclass object that implements the selected estimate technique. 

# %%
# compute the SHAP values for the linear model
background_c = shap.maskers.Independent(X, max_samples=1000) ## Concealed features may be hidden by using this function. 
explainer = shap.Explainer(l_model, background_c,
feature_names=list(X.columns))
shap_values_c = explainer(X)
shap_values = pd.DataFrame(shap_values_c.values)
shap_values.columns = list(X.columns)
shap_values

# %% [markdown]
# There is a strong, perfect linear relationship between account length and SHAP values of account length in the scatterplot. 

# %%
shap.plots.scatter(shap_values_c[:,'account_length'])

# %% [markdown]
# This shows which characteristic is more important in the classification.
# Customers who have more complaints are more likely to call customer service and can churn at any time.
# Another factor is total day in minutes, followed by number of voicemail messages. Towards the end, the seven most minor significant features are grouped.
# 
# The maximum absolute SHAP value for each feature is shown below; however, the two graphs are similar.
# A beeswarm graphic displays the SHAP value and its influence on model output.
# The heatmap display of SHAP values for hundreds of records illustrates the SHAP value density versus model features.
# The best feature has a high SHAP value, while the feature importance and SHAP value decline with time. 

# %%
# make a standard partial dependence plot
sample_ind = 25
fig,ax = shap.partial_dependence_plot(
"number_vmail_messages", m_churn_proba, X, model_expected_value=True,
feature_expected_value=True, show=False, ice=False
)
shap_values_c.feature_names
# compute the SHAP values for the linear model
explainer_log_odds = shap.Explainer(l_model, background_c,
feature_n=list(X.columns))
shap_values_churn_l = explainer_log_odds(X)
shap_values_churn_l
shap.plots.bar(shap_values_churn_l)

# %% [markdown]
# 
# The plot below illustrates how a SHAP bar plot will use the mean absolute value of each feature(by default) across all dataset occurrences (rows).

# %%
shap.plots.bar(shap_values_churn_l)

# %% [markdown]
# The below plot illustrates how utilizing the maximum absolute value highlights the number_customer_service_calls and total_intl_calls feature, which have infrequent but large magnitude impacts. 

# %%
shap.plots.bar(shap_values_churn_l.abs.max(0))

# %% [markdown]
# Beow, we can see the beeswarm plot that can be used to summarize the whole distribution of SHAP values for each feature.
# The dot's location on the x-axis indicates whether that attribute contributed positively or negatively to the prediction.
# This allows you to quickly determine if the feature is essentially flat for each forecast or significantly influences specific rows while having little impact on others.

# %%
shap.plots.beeswarm(shap_values_churn_l)

# %% [markdown]
# The below plot represents the frequency with which each feature gave SHAP values for instances utilized in the training procedure. 

# %%
shap.plots.heatmap(shap_values_churn_l[:1000])

# %% [markdown]
# ## LIME
# The SHAP values may be used to explain the logistic regression model.
# But the difficulty is time.
# With a million records, you need more time to construct all permutations and combinations to explain the local accuracy.
# LIME's explanation generation speed avoids this issue in huge dataset processing.
# 
# Explanations are the result of the LIME framework. LIME includes three primary functionalities:
# 
#     - The image explainer interprets image classification models.
#     - The text explainer gives insight into text-based models.
#     - The tabular explainer determines how much a tabular dataset's features are evaluated throughout the classification process.
# 
# 
# Lime Tabular Explainer is required to explain tabular matrix data. The term "local" refers to the framework's analysis of individual data. It does not provide a comprehensive explanation for why the model performs, but instead describes how a given observation is classified. The user should be able to grasp what a model performs if it is interpretable. Thus, while dealing with image classification, it reveals which parts of the image it evaluated when making predictions, and when working with tabular data, it shows which features influence its choice. Model-agnostic means that it may be used to any blackbox algorithm that exists now or developed in the future.
# 
# To generate a LIME output, we define the explanation as explainer.explain _instance and include the observation we picked above, the model.predict_proba, and 16 features, which show us which features are thought to be the most significant in predicting the target variable. 
# 
# The explaiser:
# 
# The explainer itself is part of the LIME library and is presented in the preceding program. Because the explainer had no default settings, we had to specify all of the parameters manually. We first call our now-formatted dataset, followed by a list of all features in our dataset.
# 
#     - X_train = Training set
#     - feature_names = Concatenated list of all feature names
#     - class_names = Target values
#     - Kernel width = Parameter to control the linearity of the induced model; the larger the width more linear is the model
# 
# 
# 
# 

# %%
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(xtrain),
feature_names=list(xtrain.columns),
class_names=['churn_dum'],
verbose=True, mode='classification')
# this record is a no churn scenario
expl = explainer.explain_instance(xtest.iloc[0], l_model.predict_proba,
num_features=16)
expl.as_list()

# %% [markdown]
# Once the explainer model object is created, you may construct explanations by checking for individual and global predictions.
# In classification with two or more classes, you can produce different feature importances for each class in relation to the features column.
# 
#  For example, total_intl_minutes had a value lower than 8.50 lowered the score of
# the model by about 0.04.

# %%
pd.DataFrame(expl.as_list())

# %% [markdown]
# Intercept 0.11796923846596004
# Prediction_local [0.10779621]
# Right: 0.1242113883724509

# %% [markdown]
# Running the code produces the LIME output divided into three sections: prediction probabilities on the left, feature probabilities in the middle, and a feature-value table on the right. A graph of prediction probabilities indicates what the model thinks will happen and the related likelihood. There is an 91% chance that the customer will not churn, which is represented by the blue bar, and a 9% chance that he will churn, which is represented by the orange bar.
# 
# The feature probability graph illustrates how much a feature impacts a specific choice. The variable number_customer_service_calls is the most influential component in this observation, and it confirms the forecast that the customer will not churn. The second most essential attribute is total_day_minutes. The last graph is the feature value table which displays the actual value of this feature in this observation. 
# 

# %%
expl.show_in_notebook(show_table=True)

# %%


# %% [markdown]
# Although LIME is simple and effective, it is not without flaws. It will be published in 2020 that the first theoretical examination of LIME confirms the importance and relevance of LIME, but it also shows that poor parameter selections might cause LIME to overlook important features. As a result, different interpretations of the same prediction may lead to deployment issues. DLIME, a deterministic variant of LIME, is suggested to overcome this uncertainty. Hierarchical clustering is used to group the data, and k-nearest neighbors (KNN) pick the cluster where the instance in question is thought to reside. 

# %% [markdown]
# ## Using SHAP for Ensemble Models
# we will use the popular Boston housing prices dataset to explain the model predictions in a regression use case scenario.
# The following are the
# variables from the Boston housing prices dataset:
# 
# CRIM: Per capita crime rate by town
# 
# • ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# • INDUS: Proportion of non-retail business acres per town
# 
# • CHAS: Charles River dummy variable (1 if tract bounds river; 0
# otherwise)
# 
# • NOX: Nitric oxide concentration (parts per 10 million)
# 
# • RM: Average number of rooms per dwelling
# 
# • AGE: Proportion of owner-occupied units built prior to 1940
# 
# • DIS: Weighted distances to five Boston employment centers
# 
# • RAD: Index of accessibility to radial highways
# 
# • TAX: Full value property tax rate per $10,000
# 
# • PTRATIO: Pupil-teacher ratio by town
# 
# • B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 
# • LSTAT: % lower status of the population
# 
# • MEDV: Median value of owner-occupied homes in $1000s

# %%
# boston Housing price 
X,y = shap.datasets.boston()
X1000 = shap.utils.sample(X, 1000) # 1000 instances for use as the background distribution
# a simple linear model
m_del = sklearn.linear_model.LinearRegression()
m_del.fit(X, y)

# %% [markdown]
# The Boston housing prices dataset is now part of the SHAP library. The base model calculation happens using the linear regression model so that you can perform the ensemble model on this dataset and compare the results.

# %%
print("coefficients of the model:\n")
for i in range(X.shape[1]):
    print(X.columns[i], "=", m_del.coef_[i].round(4))

# %% [markdown]
# The starting point is a model's coefficients.
# You'll then compare the coefficients in the complex ensemble models to those in the base linear model. Compare the explanations as well.
# Improved explainability is directly proportional to increased accuracy in prediction. 

# %%
shap.plots.partial_dependence(
"RM", m_del.predict, X1000, ice=False,
model_expected_value=True, feature_expected_value=True
)

# %% [markdown]
# You may see the predicted median value of the housing price by looking at the horizontal dotted line E[f(x)]. 
# There is a linear relationship between the RM feature and the model's predicted outcome. 

# %%
# SHAP values computation for the linear model
explainer1 = shap.Explainer(m_del.predict, X1000)
shap_values = explainer1(X)
# make a standard partial dependence plot
sample_ind = 18
shap.partial_dependence_plot(
"RM", m_del.predict, X1000, model_expected_value=True,
feature_expected_value=True, ice=False,
shap_values=shap_values[sample_ind:sample_ind+1,:]
)

# %% [markdown]
# From the above plot we can see that row number 18 from the dataset is superimposed on the PDP plot.
# RM's marginal contribution to the predicted value of the target column is illustrated by an upward-rising straight line in the picture above.
# The discrepancy between the expected value and the average predicted value is shown by the red line in the graph. 

# %%
X1000 = shap.utils.sample(X,100)
m_del.predict(X1000).mean() ## mean
m_del.predict(X1000).min() ## minimum
m_del.predict(X1000).max() ## maximum
shap_values[18:19,:] ## shap values
X[18:19]
m_del.predict(X[18:19])
shap_values[18:19,:].values.sum() + shap_values[18:19,:].base_values

# %% [markdown]
# The predicted outcome for record number 18 is 16.178, and the total of the SHAP values from various features, as well as the base value, is equal to the predicted value. 
#  From the below plot, SHAP values are generated using a linear model, which explains why the relationship is linear.
# You can expect the line to be non-linear if you switch to a non-linear model. 

# %%
shap.plots.scatter(shap_values[:,"RM"])

# %% [markdown]
# the plot below display the relationship between predicted result and SHAP values

# %%
# the waterfall_plot 
m_del.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[sample_ind], max_display=13)

# %% [markdown]
# The horizontal axis in the figure above displays the predicted result average value, which is 22.841, while the vertical axis shows the SHAP values from different features.
# The dataset's presumed values for each feature are represented in grey, while the negative SHAP values are shown in blue and the positive SHAP values are shown in red. The vertical axis also shows the predicted result for the 18th record, which is 16.178. 

# %% [markdown]
# ## Using the Interpret Explaining Boosting Model
# 
# In this section, you will utilize generalized additive models (GAM) to forecast home prices.
# The model fitted using the SHAP library may be explained.
# The interpret Python package may be used to train the generalized additive model, and the trained model object can then be sent through the SHAP model to provide explanations for the boosting models. 
# The interpret library can be installed three ways:
# 
# !pip install interpret-core
# Using the pip install method, this is done without any dependencies.
# 
# conda install -c interpretml interpret-core
# This is a distribution based on anaconda. You may install using the conda environment's terminal. 
# 
# git clone https://github.com/interpretml/interpret.git && cd interpret/scripts && make install-core
# GitHub is used to get this directly from the source.
# 
# Glassbox models: The scikit-learn framework is used to build Glassbox models that are more interpretable while keeping the same degree of accuracy as the current sklearn library. Linear models, decision trees, decision rules, and boosting-based models are all supported.
# 
# Blackbox explainers: An approximate explanation of the model's behavior and predictions is provided by blackbox explainers.
# These approaches may be used when none of the machine learning model's components can be interpreted.
# Shapely explanations, LIME explanations, partial dependency plots, and Morris sensitivity analysis may all be supported by these methods.
# 

# %% [markdown]
# To begin, import the glassbox module from interpret, then set up the explainable boosting regressor and fit the model.
# model ebm is the model object. 

# %%
# fit a GAM model to the data
m_ebm = interpret.glassbox.ExplainableBoostingRegressor()
m_ebm.fit(X, y)

# %% [markdown]
# You will sample the training dataset to provide a backdrop for creating explanations using the SHAP package.
# In the SHAP explainer, you utilize m_ebm.predict and some samples to construct explanations. 

# %%
# GAM model with SHAP explanation
expl_ebm = shap.Explainer(m_ebm.predict, X1000)
shap_v_ebm = expl_ebm(X)

# %%
# PDP with a single SHAP value 
fig,ax = shap.partial_dependence_plot(
"RM", m_ebm.predict, X, feature_expected_value=True, model_expected_value=True, show=False,ice= False,
shap_values=shap_v_ebm[sample_ind:sample_ind+1,:]
)

# %% [markdown]
# The boosting-based model is shown above .
# There is a non-linear relationship between the RM values and the forecasted target column, which is the average value of housing prices. As the red straight line indicates, we're explaining the same 18th record once again. 

# %%
shap.plots.scatter(shap_v_ebm[:,"RM"])

# %% [markdown]
# The relationship shown in the graph above is non-linear.
# At the start, the predicted value does not grow significantly as the RM increases, but beyond a particular stage, the SHAP value for RM climbs exponentially as the RM value increases. 

# %% [markdown]
# Here's another representation of the relationship between the SHAP and feature values, as seen in the figure below. 

# %%
# In order to get at explainer.expected_value, we use the waterfall_plot. 
m_ebm.predict(X)[sample_ind]
shap.plots.beeswarm(shap_v_ebm, max_display=14)

# %% [markdown]
# Non-linearity is seen in the below figure, where the extreme gradient boosting regression model is applied to explain ensemble models. 

# %%
#  XGBoost model training
m_xgb = xgboost.XGBRegressor(n_estimators=100, max_depth=2).fit(X, y)
# the GAM model explanation with SHAP
expl_xgb = shap.Explainer(m_xgb, X1000)
shap_v_xgb = expl_xgb(X)
## PDP
fig,ax = shap.partial_dependence_plot(
"RM", m_ebm.predict, X, feature_expected_value=True, model_expected_value=True, show=False,ice= False,
shap_values=shap_v_ebm[sample_ind:sample_ind+1,:]
)

# %% [markdown]
# A non-linear relationship between RM and the SHAP value of RM is seen in the figure below. 

# %%
shap.plots.scatter(shap_v_xgb[:,"RM"])

# %% [markdown]
# The graph below depicts the same non-linear relationship with an extra overlay of the RAD feature, demonstrating that the higher the RM value, the higher the RAD component, and vice versa. 

# %%
shap.plots.scatter(shap_v_xgb[:,"RM"], color=shap_values)


