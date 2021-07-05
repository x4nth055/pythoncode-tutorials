import numpy as np 
import pandas as pd 
from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours,InstanceHardnessThreshold,TomekLinks
from imblearn.over_sampling import RandomOverSampler 
import smote_variants as sv


df=pd.read_csv("creditcard.csv")
y=df["Class"]
X=df.drop(["Time","Class"],axis=1)
print(y.value_counts())


under=RandomUnderSampler()
X_und,y_und=under.fit_resample(X,y)
print(len(X_und[X_und==1])==len(X_und[X_und==0]))


over=RandomOverSampler()
X_und,y_und=over.fit_resample(X,y)
print(len(X_und[X_und==1])==len(X_und[X_und==0]))


under_samp_models=[EditedNearestNeighbours(),InstanceHardnessThreshold(),TomekLinks()]
for under_samp_model in under_samp_models:
    X_und,y_und=under_samp_model.fit_resample(X,y)
    print(X_und.shape)
    
svs=[sv.kmeans_SMOTE(),sv.Safe_Level_SMOTE(),sv.SMOTE_Cosine()]
for over_sampler in svs: 
    X_over_samp, y_over_samp= over_sampler.sample(X, y)
    print(X_over_samp.shape)