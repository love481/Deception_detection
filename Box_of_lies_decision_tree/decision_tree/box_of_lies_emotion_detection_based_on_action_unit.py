from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.tree import export_graphviz
import random
appended_data_prev=[]
# making dataframe 
for i in range(1,26):
    df = pd.read_csv(str(i)+".csv",delimiter='\t',header=None)
    H_F=df[df[0]=='Host_General face'].copy()
    H_F.drop(H_F.columns[[0, 1]], axis=1, inplace=True)
    H_E_B=df[df[0]=='Host_Eyebrows']
    H_E=df[df[0]=='Host_Eyes']
    H_G=df[df[0]=='Host_Gaze']
    H_M_O=df[df[0]=='Host_Mouth_Openess']
    H_M_L=df[df[0]=='Host_Mouth_Lips']
    G_F=df[df[0]=='Guest_General face'].copy()
    G_F.drop(G_F.columns[[0, 1]], axis=1, inplace=True)
    G_E_B=df[df[0]=='Guest_Eyebrows']
    G_E=df[df[0]=='Guest_Eyes']
    G_G=df[df[0]=='Guest_Gaze']
    G_M_O=df[df[0]=='Guest_Mouth_Openess']
    G_M_L=df[df[0]=='Guest_Mouth_Lips']
    H_F=H_F.rename(columns={2:'Start_Time',3:'End_Time',4:'Duration_s',5:'Emotions'})
    H_F.insert(3,"Eyebrows",H_E_B[5].values, True)
    H_F.insert(4,"Eyes",H_E[5].values, True)
    H_F.insert(5,"Gaze",H_G[5].values, True)
    H_F.insert(6,"Mouth_Openess",H_M_O[5].values, True)
    H_F.insert(7,"Mouth_Lips",H_M_L[5].values, True)
    G_F=G_F.rename(columns={2:'Start_Time',3:'End_Time',4:'Duration_s',5:'Emotions'})
    G_F.insert(3,"Eyebrows",G_E_B[5].values, True)                                                                                                                                                                                                                                                                                                                  
    G_F.insert(4,"Eyes",G_E[5].values, True)
    G_F.insert(5,"Gaze",G_G[5].values, True)
    G_F.insert(6,"Mouth_Openess",G_M_O[5].values, True)
    G_F.insert(7,"Mouth_Lips",G_M_L[5].values, True)
    Data = pd.concat([H_F, G_F], ignore_index=True, sort=False)
    Data.sort_values(by=['Start_Time'], inplace=True,ignore_index=True)                                                                                                                                                                                                                     
    appended_data_prev.append(Data.iloc[:,3:9])
label_encoder = LabelEncoder()
appended_data_prev = pd.concat(appended_data_prev,ignore_index=True)
appended_data_prev = appended_data_prev.mask(appended_data_prev.eq('None')).dropna()
appended_data=appended_data_prev.copy()
#print(list(appended_data.columns)[:-1])
### Eyebrows ########
# neutral/normal -> Nothings
# raising        ->  AU1, AU2
# frowning       ->  AU3
# other          ->                                                                                                                                                                                                                                                                                                                                                                                                            
# ['neutral/normal' 'raising' 'None' 'frowning' 'other']
# [2 4 0 1 3], [1 3 0 2] second list for without None, and the structure above it is same.
#####################
appended_data['Eyebrows'].replace('neutral/normal',)
appended_data['Eyebrows'] = label_encoder.fit_transform(appended_data['Eyebrows']) 
### Eyes ########
# neutral/open       -> Nothings
# Closing-both       -> AU43
# Closing-repeated   -> AU45, AU46
# Other              -> 
# Exaggerated opening-> 
# Closing-one        -> AU43_U
# ['neutral/open' 'Closing-both' 'None' 'Closing-repeated' 'Other' 'Exaggerated Opening' 'Closing-one']
# [6 0 4 2 5 3 1],[5 0 2 4 3 1]
###############################
appended_data['Eyes'] = label_encoder.fit_transform(appended_data['Eyes']) 
### Gaze ########
# Towards interlocutor -> AU69 or The AU 4, 5, or 7, alone or in combination, occurs while the eye position is fixed on the other person in the conversation.
# Towards object       -> 
# Siedeways            -> AU61, AU62
# Other                -> 
# Down                 -> AU64
# Up                   -> AU63
# Towards audience     -> 
# ['Towards interlocutor' 'Towards object' 'Siedeways' 'Other' 'Down' 'Up' 'None' 'Towards audience']
# [5 6 3 2 0 7 1 4],[4 5 2 1 0 6 3]
#################
appended_data['Gaze'] = label_encoder.fit_transform(appended_data['Gaze']) 
### Mouth_Openess ########
# ['Open mouth' 'Closed mouth' 'Other' 'None']
# Open mouth   -> AU27
# Closed mouth -> 
# Other        ->
# [2 0 3 1], [1 0 2]
####################
appended_data['Mouth_Openess'] = label_encoder.fit_transform(appended_data['Mouth_Openess']) 
## Mouth_Lips #########
# ['neutral' 'Corners up' 'Retracted' 'Other' 'Corners down' 'Protruded' 'None']
# neutral      -> nothings
# Corners up   -> AU12
# Retracted    ->  AU25, AU26
# Other        ->
# Corners down -> AU15
# Protruded    -> AU18, AU22, AU25
# [6 1 5 3 0 4 2],[5 1 4 2 0 3]
########################
appended_data['Mouth_Lips'] = label_encoder.fit_transform(appended_data['Mouth_Lips']) 
### Emotions #########
# ['neutral' 'other' 'smile' 'laughter' 'scowl' 'None']
# [2 3 5 1 4 0],[1 2 4 0 3]
######################
appended_data['Emotions'] = label_encoder.fit_transform(appended_data['Emotions'])
#print(list(appended_data['Emotions'].unique())) 
X=appended_data[['Eyebrows', 'Eyes', 'Gaze', 'Mouth_Openess','Mouth_Lips']]
Y=appended_data['Emotions']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
dtree = DecisionTreeClassifier(max_depth=3, criterion='gini')
dtree.fit(X_train, y_train)
print(X,Y)
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(dtree, 
                   feature_names=list(appended_data_prev.columns)[:-1],
                   class_names=list(appended_data_prev['Emotions'].unique()),
                   filled=True)
fig.savefig("decision_tree.png")
predictions = dtree.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)
        
         
