# %%
!gdown --id 12vfq3DYFId3bsXuNj_PhsACMzrLTfObs

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

# %%
#reading data
data = pd.read_csv("data_regression.csv")
##The dimension of the data is seen, and the output column is checked to see whether it is continuous or discrete. 
##In this case, the output is discrete, so a classification algorithm should be applied.
data = data.drop(["year", "customer_id", "phone_no"], axis=1)
print(data.shape)         # Lookiing the shape of the data
print(data.columns)       # Looking how many columns data has
data.dtypes  
data.head()

# %%
data.isnull().sum()

# %%
final_data = data.dropna()         # Dropping the null values
final_data.head()

# %%
final_data["churn"].value_counts()       
# let us see how many data is there in each class for deciding the sampling data number

# %%
data_majority = final_data[final_data['churn']==0] # class 0
data_minority = final_data[final_data['churn']==1] # class 1
# upsampling minority class
data_minority_upsampled = resample(data_minority, replace=True, n_samples=900, random_state=123) 
# downsampling majority class
data_majority_downsampled = resample(data_majority, replace=False, n_samples=900, random_state=123)
# concanating both upsampled and downsampled class
## Data Concatenation:  Concatenating the dataframe after upsampling and downsampling 
# concanating both upsampled and downsampled class
data2 = pd.concat([data_majority_downsampled, data_minority_upsampled])
## Encoding Catagoricals:  We need to encode the categorical variables before feeding it to the model
data2[['gender', 'multi_screen', 'mail_subscribed']]
# label encoding categorical variables
label_encoder = preprocessing.LabelEncoder()
data2['gender'] = label_encoder.fit_transform(data2['gender'])
data2['multi_screen'] = label_encoder.fit_transform(data2['multi_screen'])
data2['mail_subscribed'] = label_encoder.fit_transform(data2['mail_subscribed'])
## Lets now check again the distribution of the oputut class after sampling
data2["churn"].value_counts()

# %%
# indenpendent variable 
X = data2.iloc[:,:-1]
## This X will be fed to the model to learn params 
#scaling the data
sc = StandardScaler()         # Bringing the mean to 0 and variance to 1, so as to have a non-noisy optimization
X = sc.fit_transform(X)
X = sc.transform(X)
## Keeping the output column in a separate dataframe
data2 = data2.sample(frac=1).reset_index(drop=True) ## Shuffle the data frame and reset index
n_samples, n_features = X.shape ## n_samples is the number of samples and n_features is the number of features
#output column
Y = data2["churn"]
#output column
Y = data2["churn"]
##Data Splitting: 
## The data is processed, so now we can split the data into train and test to train the model with training data and test it later from testing data.
#splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify = Y)
print((y_train == 1).sum())
print((y_train == 0).sum())

# %%
print(type(X_train))
print(type(X_test))
print(type(y_train.values))
print(type(y_test.values))

# %%
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.values.astype(np.float32))
y_test = torch.from_numpy(y_test.values.astype(np.float32))

# %%
y_train.shape, y_test.shape

# %%
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# %%
y_train.shape, y_test.shape

# %%
# logistic regression class
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    #sigmoid transformation of the input 
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# %%
lr = LogisticRegression(n_features)

# %%
num_epochs = 500
# Traning the model for large number of epochs to see better results  
learning_rate = 0.0001
criterion = nn.BCELoss()                                
# We are working on lgistic regression so using Binary Cross Entropy
optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)      

# %%
for epoch in range(num_epochs):
    y_pred = lr(X_train)
    loss = criterion(y_pred, y_train)             
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 20 == 0:                                         
        # printing loss values on every 10 epochs to keep track
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# %%
with torch.no_grad():
    y_predicted = lr(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')

# %%
#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted_cls))

# %%
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predicted_cls)
print(confusion_matrix)

# %%



