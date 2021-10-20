#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt              
import seaborn as sns                   


# In[2]:


pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)  
df=pd.read_csv("E:/Study/ML tuts/Case Studies/fifa/players_20_classification.csv")
df


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.drop(
['sofifa_id',
'player_url',
'short_name',
'long_name',
'dob',
'preferred_foot',
'player_tags',
'player_traits',
'player_positions',
'nation_jersey_number',
'team_jersey_number',
'pace',         
'shooting',      
'passing',       
'dribbling',    
'defending',     
'physique',      
'gk_diving',     
'gk_handling',   
'gk_kicking',    
'gk_reflexes',   
'gk_speed',      
'gk_positioning',
'overall'
],axis=1,inplace=True)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.nunique()


# In[10]:


df["age"].unique()


# In[11]:


df["height_cm"].unique()


# In[12]:


df["weight_kg"].unique()


# In[13]:


df["nationality"].unique()


# In[14]:


print(df['nationality'].isnull().values.any()) 


# In[15]:


df["club"].unique()


# In[16]:


print(df['club'].isnull().values.any()) 


# In[17]:


df["international_reputation"].unique()


# In[18]:


df["weak_foot"].unique()


# In[19]:


df["skill_moves"].unique()


# In[20]:


df["work_rate"].unique()


# In[21]:


df["body_type"].unique()


# In[22]:


df["player_type"].unique()


# In[23]:


df['attacking_crossing'].unique()


# In[24]:


df['attacking_finishing'].unique()


# In[25]:


df['attacking_heading_accuracy'].unique()


# In[26]:


df['attacking_short_passing'].unique()


# In[27]:


df['attacking_volleys'].unique()


# In[28]:


df['skill_dribbling'].unique()


# In[29]:


df['skill_curve'].unique()


# In[30]:


df['skill_fk_accuracy'].unique()


# In[31]:


df['skill_long_passing'].unique()


# In[32]:


df['skill_ball_control'].unique()


# In[33]:


df['movement_acceleration'].unique()


# In[34]:


df['movement_sprint_speed'].unique()


# In[35]:


df['movement_agility'].unique()


# In[36]:


df['movement_reactions'].unique()


# In[37]:


df['movement_balance'].unique()


# In[38]:


df['power_shot_power'].unique()


# In[39]:


df['power_jumping'].unique()


# In[40]:


df['power_stamina'].unique()


# In[41]:


df['power_strength'].unique()


# In[42]:


df['power_long_shots'].unique()


# In[43]:


df['mentality_aggression'].unique()


# In[44]:


df['mentality_interceptions'].unique()


# In[45]:


df['mentality_positioning'].unique()


# In[46]:


df['mentality_vision'].unique()


# In[47]:


df['mentality_penalties'].unique()


# In[48]:


df['mentality_composure'].unique()


# In[49]:


df['defending_marking'].unique()


# In[50]:


df['defending_standing_tackle'].unique()


# In[51]:


df['defending_sliding_tackle'].unique()


# In[52]:


df['goalkeeping_diving'].unique()


# In[53]:


df['goalkeeping_handling'].unique()


# In[54]:


df['goalkeeping_kicking'].unique()


# In[55]:


df['goalkeeping_positioning'].unique()


# In[56]:


df['goalkeeping_reflexes'].unique()


# In[57]:


df.info()


# In[58]:


df_cat=df.select_dtypes(object)  
df_num=df.select_dtypes(['int64']) 


# In[59]:


df_num.head()


# In[60]:


df_cat.head()


# In[61]:


from sklearn.preprocessing import LabelEncoder


# In[62]:


for col in df_cat:
    le=LabelEncoder()
    df_cat[col]=le.fit_transform(df_cat[col])
print(df_cat.head())


# In[63]:


df_cat.nunique()


# In[64]:


new_df=pd.concat([df_num,df_cat],axis=1)
new_df.head()


# In[65]:


new_df.info()


# In[66]:


df['player_type'].value_counts()


# In[67]:


sns.countplot(new_df['player_type'])
plt.show()


# From above, we can see that the player_type has been encoded as :
# 
#         0 - Attacker
#         1 - Defender
#         2 - Goalkeeper
#         3 - Midfielder

# In[68]:


sns.countplot(new_df['player_type'],hue=new_df['body_type'])
plt.show()


# # Implementing Logistic Regression

# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


x=new_df.drop('player_type',axis=1) #considering all independent variables
y=new_df['player_type']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


logr=LogisticRegression()
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)    # will provide the prediction in binary classification for each row present in x_test
print(y_pred)


# In[73]:


np.set_printoptions(threshold=sys.maxsize)


# In[74]:


print(y_pred)


# In[75]:


y_predicted=logr.predict_proba(x_test)
print(y_predicted)


# In[76]:


print("Intercept : ",logr.intercept_)
print("Slope : ",logr.coef_)


# In[77]:


logr.score(x_test,y_test) #Returns the mean accuracy on the given test data and labels.


# # Confusion Matrix for multiple classes in target variable

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[78]:


from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix


# In[79]:


confusion=confusion_matrix(y_test,y_pred)


# In[80]:


print('Confusion Matrix :\n')
print(confusion)


# In[81]:


sns.countplot(y_test)
plt.show()


# In[82]:


multilabel_confusion_matrix(y_test, y_pred,labels=[0,1,2,3])


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[83]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[84]:


accuracy_score(y_test,y_pred)


# In[85]:


recall_score(y_test,y_pred,average='macro')


# In[86]:


precision_score(y_test,y_pred,average='macro')


# In[87]:


f1_score(y_test,y_pred,average='macro')


# In[88]:


from sklearn.metrics import roc_auc_score


# In[89]:


roc_auc_score(y_test,y_predicted,average='macro',multi_class='ovr')  

#We have used "y_predicted=logr.predict_proba(x_test)"
#as y_pred gave values (0,1,2,3) and using y_pred gave error "AxisError: axis 1 is out of bounds for array of dimension 1"
#y_pred is used in "Multilabel" classification whereas y_predicted is used in "Multiclass" classification in above function.


# Multiclass classification : a fruit can be either an apple or a pear but not both at the same time.
#     
# Multilabel classification : A Document might be about any of religion, politics, finance or education 
#                             at the same time or none of these.

# In[90]:


from sklearn.metrics import roc_curve


# In[91]:


#fpr,tpr,threshold=roc_curve(y_test,y_pred)  
#Note: The above function is restricted to ONLY binary classification task.


# # Implementing Decision Tree

# In[92]:


sns.countplot(new_df['player_type'])
plt.show()


# In[93]:


from sklearn.tree import DecisionTreeClassifier


# In[94]:


from sklearn import tree    #importing tree to draw the decision tree
fig = plt.gcf()
fig.set_size_inches(150, 100)


# In[95]:


dt=DecisionTreeClassifier()    # default criterion = GINI index
dt.fit(x_train,y_train)
print("Decision Tree Score : ",dt.score(x_test,y_test))
print("Bias for Entropy : ",dt.score(x_train,y_train))


# In[96]:


tree.plot_tree(dt.fit(x_train,y_train),fontsize=6)


# In[97]:


dt1=DecisionTreeClassifier(criterion="entropy")    # using criterion as Entropy
dt1.fit(x_train,y_train)
print("Decision Tree Score : ",dt1.score(x_test,y_test))
print("Bias for Entropy : ",dt1.score(x_train,y_train))


# In[98]:


tree.plot_tree(dt1.fit(x_train,y_train),fontsize=6)


# # Performing Pruning for both Gini and Entropy Decision Trees using max_depth

# In[99]:


dt2=DecisionTreeClassifier(criterion="gini",max_depth=10)    # using criterion as gini and max_depth as 4
dt2.fit(x_train,y_train)
print("Decision Tree Score : ",dt2.score(x_test,y_test))
print("Bias : ",dt2.score(x_train,y_train))


# In[100]:


tree.plot_tree(dt2.fit(x_train,y_train),fontsize=8)


# In[101]:


dt3=DecisionTreeClassifier(criterion="entropy",max_depth=10)    # using criterion as entropy and max_depth as 7
dt3.fit(x_train,y_train)
print("Decision Tree Score : ",dt3.score(x_test,y_test))
print("Bias : ",dt3.score(x_train,y_train))


# In[102]:


tree.plot_tree(dt3.fit(x_train,y_train),fontsize=8)


# # Implementing ANOVA Feature Selection for Classification

# In[103]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# In[104]:


anova=SelectKBest(score_func=f_regression,k=22)
anova.fit(x_train,y_train)
x_train_anova=anova.transform(x_train)
x_test_anova=anova.transform(x_test)


# In[105]:


scores_df = pd.DataFrame(anova.scores_)
columns_df = pd.DataFrame(x.columns)
featureScore_anova = pd.concat([columns_df, scores_df], axis=1)
featureScore_anova.columns = ['Feature Name', 'Score']
featureScore_anova


# In[106]:


featureScore_anova.sort_values("Score", ascending=False)


# In[107]:


featureScore_anova.nlargest(22, "Score")


# In[108]:


from sklearn.linear_model import LogisticRegression  


# In[109]:


lr=LogisticRegression()
lr.fit(x_train_anova,y_train)
print("Bias of Logistic Regression after Annova test = ",lr.score(x_train_anova,y_train))
print("Variance of Logistic Regression after Annova test =",lr.score(x_test_anova,y_test))


# # Implementing K-Nearest Neighbours

# In[110]:


from sklearn.neighbors import KNeighborsClassifier


# In[111]:


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_test,y_test) 


# In[112]:


y_pred=knn.predict(x_test)   #to check probability scores received after classification of training data
y_pred


# In[113]:


tbl=pd.DataFrame(list(zip(y_test,y_pred)),columns=['y_observed','y_predicted'])
print(tbl)


# # Conclusion :

# From the above, we can conclude that Logistic Regression model built using important features to classify samples as either 
# 
# "Attacker"/"Midfielder"/"Defender"/"Goalkeeper" gave us the most highest accruracy/recall/precision/f1 scores 
# 
# and therefore should be considered as the final classification model.
