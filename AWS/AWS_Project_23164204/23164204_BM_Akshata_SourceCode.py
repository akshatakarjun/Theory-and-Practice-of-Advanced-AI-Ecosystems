#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[93]:


data = pd.read_csv('Final_Encoded_Travel_Details.csv', header = None)


# In[94]:


data.head()


# In[76]:


print(data.isnull().sum())


# In[95]:


df = data.drop([1,3,4,6],axis=1)


# In[96]:


df.head()


# In[13]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


# In[14]:


df = df.sample(frac=1).reset_index(drop=True)


# In[97]:


train, test_and_validate = train_test_split(df,
                                            test_size=0.2,
                                            random_state=42
                                            )


# In[98]:


test, validate = train_test_split(test_and_validate,
                                  test_size=0.5,
                                  random_state=42
                                  )


# In[99]:


print(train.shape)
print(test.shape)
print(validate.shape)


# In[105]:


import boto3

bucket_name = 'hotel.management'

train.to_csv('training_data.csv', header = None, index = False)
key = 'data/train/training_data'
url = 's3://{}/{}'.format(bucket_name,key)
boto3.Session().resource('s3').Bucket(bucket_name).Object(key).upload_file('training_data.csv')


# In[106]:


validate.to_csv('validating_data.csv', header = None, index = False)
key = 'data/validate/validating_data'
url = 's3://{}/{}'.format(bucket_name,key)
boto3.Session().resource('s3').Bucket(bucket_name).Object(key).upload_file('validating_data.csv')

test.to_csv('testing_data.csv', header = None, index = False)
key = 'data/test/testing_data'
url = 's3://{}/{}'.format(bucket_name,key)
boto3.Session().resource('s3').Bucket(bucket_name).Object(key).upload_file('testing_data.csv')


# In[102]:


import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker import get_execution_role

key = 'model/xgb_model'
s3_output_location = url = 's3://{}/{}'.format(bucket_name, key)

xgb_model = sagemaker.estimator.Estimator(
    get_image_uri(boto3.Session().region_name, 'xgboost'),
    get_execution_role(),
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    train_volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session()
)
xgb_model.set_hyperparameters(max_depth=5,
                              eta=0.2,
                              gamma=4,
                              min_child_weight=6,
                              silent=0,
                              objective='multi:softmax',
                              num_class=7,
                              num_round=5)


# In[103]:


train_data='s3://{}/{}'.format(bucket_name,'data/train')
validate_data='s3://{}/{}'.format(bucket_name,'data/validate')
                                  
train_channel=sagemaker.session.s3_input(train_data, content_type='text/csv')
validate_channel=sagemaker.session.s3_input(validate_data, content_type='text/csv')
                            
data_channels = {'train':train_channel,'validation':validate_channel}
                                  
xgb_model.fit(inputs=data_channels)


# In[104]:


xgb_predictor = xgb_model.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')


# In[ ]:




