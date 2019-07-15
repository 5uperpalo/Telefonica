
# 1. Define libraries datasets, scaling


```python
# define libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# https://xgboost.readthedocs.io/en/latest/
import xgboost
# https://scikit-learn.org/stable/modules/svm.html
from sklearn import svm
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# defining scoring strategy:
# https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
# scoring needs to be changed with string, ie : LogisticRegressionCV(cv=10, random_state=0,multi_class='multinomial', scoring="f1_score").fit(samples, labels)
# https://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.model_selection import cross_val_score


#sklearn.svm.LinearSVC (setting multi_class=”crammer_singer”)
#sklearn.linear_model.LogisticRegression (setting multi_class=”multinomial”)
#sklearn.linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)
```


```python
# define datasets
antenna_dt_loc = '/home/sop/LondonMobility/results/Features/SectorPerf/Summary/AntennaMLDataset_FebMar_df.csv'
residents_dt_loc = '/home/sop/LondonMobility/results/Features/Performance/Summary/part-00000-99a02b3d-d2ec-41c8-a02f-f8acbc8f9c5b-c000.csv'

# drop rows with empty values
antenna_dt_pd = pd.read_csv(antenna_dt_loc, index_col=False)
antenna_dt_pd.dropna(inplace=True)

residents_dt_pd = pd.read_csv(residents_dt_loc, index_col=False)
residents_dt_pd.dropna(inplace=True)
```


```python
# drop lkey and device_id columns as they are not needed for machine learning
antenna_dt_pd_ml = antenna_dt_pd.drop(['lkey'], axis=1)
residents_dt_pd_ml = residents_dt_pd.drop(['device_id'], axis=1)

# scale the dataset to get better results, scale everything except LSOA_IMD_decile - i.e. labels
scaler = StandardScaler()
scaler.fit(antenna_dt_pd_ml[antenna_dt_pd_ml.columns[:-1]])
antenna_dt_pd_ml_scaled = pd.DataFrame(scaler.transform(antenna_dt_pd_ml[antenna_dt_pd_ml.columns[:-1]]),columns=antenna_dt_pd_ml.columns[:-1])
antenna_dt_pd_ml_scaled['LSOA_IMD_decile'] = antenna_dt_pd_ml['LSOA_IMD_decile'].values
antenna_samples = antenna_dt_pd_ml_scaled[antenna_dt_pd_ml_scaled.columns[:-1]].values
antenna_labels = antenna_dt_pd_ml_scaled['LSOA_IMD_decile'].values

scaler = StandardScaler()
scaler.fit(residents_dt_pd_ml[residents_dt_pd_ml.columns[:-1]])
residents_dt_pd_ml_scaled = pd.DataFrame(scaler.transform(residents_dt_pd_ml[residents_dt_pd_ml.columns[:-1]]),columns=residents_dt_pd_ml.columns[:-1])
residents_dt_pd_ml_scaled['LSOA_IMD_decile'] = residents_dt_pd_ml['LSOA_IMD_decile'].values
residents_samples = antenna_dt_pd_ml_scaled[antenna_dt_pd_ml_scaled.columns[:-1]].values
residents_labels = antenna_dt_pd_ml_scaled['LSOA_IMD_decile'].values
```

# 2. Graphical analysis of features in each LSOA IMD decile - boxplots of scaled features


```python
antenna_boxplot = antenna_dt_pd_ml_scaled.groupby(['LSOA_IMD_decile']).boxplot(figsize=(30, 18), subplots=True, rot=90, grid=True, layout=(10,1), sharex=True, sharey=False, showfliers=False)
title = plt.suptitle("Scaled Antenna dataset feature boxplot per LSOA IMD decile");
xlabel = plt.xlabel("features", labelpad=14)
ylabel = plt.ylabel("scaled value of features", labelpad=14)
```


![png](output_5_0.png)



```python
residents_boxplot = residents_dt_pd_ml_scaled.groupby(['LSOA_IMD_decile']).boxplot(figsize=(30, 18), subplots=True, rot=90, grid=True, layout=(10,1), sharex=True, sharey=False, showfliers=False)
title = plt.suptitle("Scaled Residents dataset feature boxplot per LSOA IMD decile");
xlabel = plt.xlabel("features", labelpad=14)
ylabel = plt.ylabel("scaled value of features", labelpad=14)
```


![png](output_6_0.png)


# 3. Sizes of clusters, i.e. number of samples per LSOA IMD decile - bar charts


```python
antenna_dt_pd_ml_scaled['LSOA_IMD_decile'].value_counts(sort=False,normalize=True).plot(kind='bar')
title = plt.title("Antenna LSOA IMD decile cluster sizes");
xlabel = plt.xlabel("LSOA IMD decile", labelpad=14)
ylabel = plt.ylabel("Fraction of all samples", labelpad=14)
```


![png](output_8_0.png)



```python
residents_dt_pd_ml_scaled['LSOA_IMD_decile'].value_counts(sort=False,normalize=True).plot(kind='bar')
title = plt.title("Residents LSOA IMD decile cluster sizes");
xlabel = plt.xlabel("LSOA IMD decile", labelpad=14)
ylabel = plt.ylabel("Fraction of all samples", labelpad=14)
```


![png](output_9_0.png)


# 4. Supervised algorithm results
## 4.1 Full feature set


```python
# XGBOOST importance types
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

# prepare labels, samples, train/test datasets
antenna_dt_pd_ml_scaled = pd.read_csv('antenna_dt_pd_ml_scaled.csv', index_col=False)
residents_dt_pd_ml_scaled = pd.read_csv('residents_dt_pd_ml_scaled.csv', index_col=False)

antenna_samples = antenna_dt_pd_ml_scaled[antenna_dt_pd_ml_scaled.columns[:-1]].values
antenna_labels = antenna_dt_pd_ml_scaled['LSOA_IMD_decile'].values

residents_samples = residents_dt_pd_ml_scaled[residents_dt_pd_ml_scaled.columns[:-1]].values
residents_labels = residents_dt_pd_ml_scaled['LSOA_IMD_decile'].values

antenna_samples_train, antenna_samples_test, antenna_labels_train, antenna_labels_test = train_test_split(antenna_samples, antenna_labels, test_size=0.1)
residents_samples_train, residents_samples_test, residents_labels_train, residents_labels_test = train_test_split(residents_samples, residents_labels, test_size=0.1)
```


```python
# logistic regression
# For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# solver = ?
# antenna
LR_clf_ant = LogisticRegression(multi_class='multinomial', solver='lbfgs')
LRscores_ant = cross_val_score(LR_clf_ant, antenna_samples, antenna_labels, cv=10, scoring='f1_weighted')
LR_clf_ant.fit(antenna_samples_train,antenna_labels_train)
LRantenna_predicted = LR_clf_ant.predict(antenna_samples_test)
print('Antenna LR 10CV f1_weighted scores : ' + str(LRscores_ant))
print('Antenna LR classification report :\n' + str(classification_report(antenna_labels_test, LRantenna_predicted)))
print('Antenna LR confusion matrix :\n' + str(confusion_matrix(antenna_labels_test, LRantenna_predicted)))
# residents
LR_clf_res = LogisticRegression(multi_class='multinomial', solver='lbfgs')
LRscores_res = cross_val_score(LR_clf_res, residents_samples, residents_labels, cv=10, scoring='f1_weighted')
LR_clf_res.fit(residents_samples_train,residents_labels_train)
LRresidents_predicted = LR_clf_res.predict(residents_samples_test)
print('Residents LR 10CV f1_weighted scores : ' + str(LRscores_res))
print('Residents LR classification report :\n' + str(classification_report(residents_labels_test, LRresidents_predicted)))
print('Residents LR confusion matrix :\n' + str(confusion_matrix(residents_labels_test, LRresidents_predicted)))

# SVM
# about gamma='scale' issue : https://stackoverflow.com/questions/52582796/support-vector-regression-typeerror-must-be-real-number-not-str
# antenna
SVM_clf_ant = svm.LinearSVC()
SVMscores_ant = cross_val_score(SVM_clf_ant, antenna_samples, antenna_labels, cv=10, scoring='f1_weighted')
SVM_clf_ant.fit(antenna_samples_train,antenna_labels_train)
SVMantenna_predicted = SVM_clf_ant.predict(antenna_samples_test)
print('Antenna SVM 10CV f1_weighted scores : ' + str(SVMscores_ant))
print('Antenna SVM classification report :\n' + str(classification_report(antenna_labels_test, SVMantenna_predicted)))
print('Antenna SVM confusion matrix :\n' + str(confusion_matrix(antenna_labels_test, SVMantenna_predicted)))
# residents
SVM_clf_res = svm.LinearSVC()
SVMscores_res = cross_val_score(SVM_clf_res, residents_samples, residents_labels, cv=10, scoring='f1_weighted')
SVM_clf_res.fit(residents_samples_train,residents_labels_train)
SVMresidents_predicted = SVM_clf_res.predict(residents_samples_test)
print('Residents SVM 10CV f1_weighted scores : ' + str(SVMscores_res))
print('Residents SVM classification report :\n' + str(classification_report(residents_labels_test, SVMresidents_predicted)))
print('Residents SVM confusion matrix :\n' + str(confusion_matrix(residents_labels_test, SVMresidents_predicted)))

# xgboost
# antenna
XGBOOST_clf_ant = xgboost.XGBClassifier()
XGBOOSTscores_ant = cross_val_score(XGBOOST_clf_ant, antenna_samples, antenna_labels, cv=10, scoring='f1_weighted')
XGBOOST_clf_ant.fit(antenna_samples_train,antenna_labels_train)
XGBOOSTantenna_predicted = XGBOOST_clf_ant.predict(antenna_samples_test)
print('Antenna XGBOOST 10CV f1_weighted scores : ' + str(XGBOOSTscores_ant))
print('Antenna XGBOOST classification report :\n' + str(classification_report(antenna_labels_test, XGBOOSTantenna_predicted)))
print('Antenna XGBOOST confusion matrix :\n' + str(confusion_matrix(antenna_labels_test, XGBOOSTantenna_predicted)))
print('Antenna XGBOOST features importances :\n' + str(XGBOOST_clf_ant.feature_importances_))

for imp_t in importance_types:
    print(imp_t + " : ",end = '')
    for f in features_names[:len(antenna_dt_pd_ml_scaled.columns[:-1])]:
        print(XGBOOST_clf_ant.get_booster().get_score(importance_type=imp_t).get(f), end = ' , ')
    print()
# residents
XGBOOST_clf_res = xgboost.XGBClassifier()
XGBOOSTscores_res = cross_val_score(XGBOOST_clf_res, residents_samples, residents_labels, cv=10, scoring='f1_weighted')
XGBOOST_clf_res.fit(residents_samples_train,residents_labels_train)
XGBOOSTresidents_predicted = XGBOOST_clf_res.predict(residents_samples_test)
print('Residents XGBOOST 10CV f1_weighted scores : ' + str(XGBOOSTscores_res))
print('Residents XGBOOST classification report :\n' + str(classification_report(residents_labels_test, XGBOOSTresidents_predicted)))
print('Residents XGBOOST confusion matrix :\n' + str(confusion_matrix(residents_labels_test, XGBOOSTresidents_predicted)))
print('Residents XGBOOST features importances :\n' + str(XGBOOST_clf_res.feature_importances_))

for imp_t in importance_types:
    print(imp_t + " : ",end = '')
    for f in features_names[:len(residents_dt_pd_ml_scaled.columns[:-1])]:
        print(XGBOOST_clf_res.get_booster().get_score(importance_type=imp_t).get(f), end = ' , ')
    print()
```

### Output of the script above
The script was run in the background on the server as the residents dataset calculations take serveral hours.
Script:

    /home/sop/LondonMobility/ml_lsoa_analysis/full_feature_set_script.py

Results:

    /home/sop/LondonMobility/ml_lsoa_analysis/full_feature_set_results.txt

#### Logistic Regression
##### Antenna
Antenna LR 10CV f1_weighted scores : 

    [0.87560556, 0.86190394, 0.8790403, , 0.87427327, 0.88034924, 0.8714265, 0.87127021, 0.87174524, 0.87368437, 0.88281454]

Antenna LR classification report :

             precision    recall  f1-score   support
          1       0.96      0.82      0.89       165
          2       0.93      0.90      0.91       448
          3       0.74      0.91      0.82       606
          4       0.87      0.87      0.87       521
          5       0.95      0.86      0.90       384
          6       0.97      0.84      0.90       353
          7       0.87      0.88      0.87       354
          8       0.82      0.86      0.84       286
          9       0.90      0.79      0.84       204
         10       0.96      0.91      0.94       104
    avg / total   0.88      0.87      0.87      3425

Antenna LR confusion matrix :

    [[136   5  16   2   0   0   1   4   1   0]
     [  2 402  19   6   3   2   3   8   3   0]
     [  2   9 554  10   2   1   8  18   2   0]
     [  0   6  46 451   4   2   6   6   0   0]
     [  0   6  23  13 329   0   7   4   2   0]
     [  1   0  28  15   4 295   5   3   2   0]
     [  0   4  20  12   0   1 311   2   2   2]
     [  0   1  18   4   3   2   4 247   6   1]
     [  0   0  17   6   0   1  10   8 161   1]
     [  0   0   4   0   0   0   3   2   0  95]]
     
##### Residents     
Residents LR 10CV f1_weighted scores : 

    [0.96798743, 0.96579157, 0.96786503, 0.96762216, 0.96658296, 0.96084779, 0.96096328, 0.96272285, 0.96168202, 0.96286898]
    
Residents LR classification report :

             precision    recall  f1-score   support
          1       0.96      0.96      0.96      2061
          2       0.97      0.97      0.97      5130
          3       0.97      0.97      0.97      6290
          4       0.97      0.96      0.97      6501
          5       0.96      0.96      0.96      4230
          6       0.96      0.96      0.96      4074
          7       0.96      0.95      0.95      4033
          8       0.96      0.96      0.96      4155
          9       0.96      0.97      0.97      3284
         10       0.97      0.97      0.97      1751
    avg / total   0.96      0.96      0.96     41509

Residents LR confusion matrix :

    [[1977   18   20   19    7    6    3    5    5    1]
     [  20 4983   47   20   28   11   12    5    3    1]
     [  17   44 6114   45   27   17   10    9    6    1]
     [  17   31   40 6262   36   31   25   36   22    1]
     [   6   25   24   26 4068   28   23   19    9    2]
     [   5    9   15   24   28 3931   15   22   17    8]
     [   6   20   29   38   30   20 3813   36   31   10]
     [   2    7   20   27   11   13   36 4002   20   17]
     [   2    1    5   10    9   14   14   23 3189   17]
     [   0    1    0    5    6    8    8   15   15 1693]]

#### SVM
##### Antenna
Antenna SVM 10CV f1_weighted scores : 
    
    [0.76111178, 0.75934491, 0.77520498, 0.76847917, 0.76965818, 0.76664688, 0.76751445, 0.78620269, 0.76955391, 0.77925448]
    
Antenna SVM classification report :

             precision    recall  f1-score   support
          1       0.93      0.73      0.82       165
          2       0.88      0.81      0.85       448
          3       0.63      0.88      0.74       606
          4       0.69      0.78      0.73       521
          5       0.86      0.71      0.78       384
          6       0.90      0.66      0.76       353
          7       0.75      0.78      0.77       354
          8       0.81      0.77      0.79       286
          9       0.89      0.65      0.75       204
         10       0.92      0.79      0.85       104
    avg / total   0.79      0.77      0.77      3425

Antenna SVM confusion matrix :

    [[120   5  30   4   1   0   1   1   3   0]
     [  0 364  38  22   5   4   8   5   2   0]
     [  5   9 535  25   6   5  11   9   1   0]
     [  3  15  65 404   8   4  13   6   3   0]
     [  0   6  44  37 272   3  18   3   0   1]
     [  1   1  48  37  11 233  12   8   0   2]
     [  0   6  28  23   3   4 277   6   5   2]
     [  0   4  30  17   4   1   8 220   1   1]
     [  0   3  24  11   4   4  15   9 133   1]
     [  0   0   5   7   1   0   4   3   2  82]]

##### Residents       
Residents SVM 10CV f1_weighted scores : 

    [0.96210845, 0.95974829, 0.96153084, 0.96160187, 0.96147266, 0.9553046, 0.95536256, 0.95855823, 0.9560232, 0.95766261]
    
Residents SVM classification report :

             precision    recall  f1-score   support
          1       0.96      0.96      0.96      2061
          2       0.96      0.97      0.96      5130
          3       0.96      0.97      0.96      6290
          4       0.95      0.96      0.96      6501
          5       0.96      0.96      0.96      4230
          6       0.96      0.96      0.96      4074
          7       0.96      0.94      0.95      4033
          8       0.96      0.96      0.96      4155
          9       0.96      0.96      0.96      3284
         10       0.96      0.96      0.96      1751
    avg / total   0.96      0.96      0.96     41509

Residents SVM confusion matrix :

    [[1975   16   21   23    7    3    4    5    5    2]
     [  23 4954   52   30   29   12   15    9    4    2]
     [  20   53 6080   61   29   16   11   11    8    1]
     [  18   32   47 6245   36   33   28   39   19    4]
     [   8   29   32   40 4043   26   22   19    8    3]
     [   6   18   28   33   24 3902   17   21   16    9]
     [   7   22   37   53   34   25 3773   38   29   15]
     [   2   14   26   35   11   11   26 3993   21   16]
     [   3    6    9   23    8   13   15   21 3168   18]
     [   0    1    5   14    5    4    6   13   15 1688]]

#### XGBOOST
##### Antenna
Antenna XGBOOST 10CV f1_weighted scores : 

    [0.9123163, , 0.8990322, , 0.90395565, 0.90791748, 0.91131416, 0.91159137, 0.91701889, 0.90934044, 0.90627642, 0.92199817]
    
Antenna XGBOOST classification report :

             precision    recall  f1-score   support
          1       0.97      0.90      0.93       165
          2       0.89      0.95      0.92       448
          3       0.87      0.91      0.89       606
          4       0.88      0.89      0.89       521
          5       0.95      0.89      0.92       384
          6       0.92      0.90      0.91       353
          7       0.92      0.91      0.91       354
          8       0.91      0.92      0.92       286
          9       0.94      0.87      0.90       204
         10       0.97      0.94      0.96       104
    avg / total   0.91      0.91      0.91      3425

Antenna XGBOOST confusion matrix :

    [[148   3  10   3   0   1   0   0   0   0]
     [  3 426   9   6   1   2   1   0   0   0]
     [  0  15 554  15   4   3   7   5   3   0]
     [  1   8  26 463   6   6   5   6   0   0]
     [  0   5   7  12 341   8   4   5   2   0]
     [  0   2  12  11   2 318   2   2   2   2]
     [  0   3  12   4   2   4 323   2   3   1]
     [  0   9   4   3   1   1   4 263   1   0]
     [  0   5   1   7   1   3   6   3 178   0]
     [  0   0   2   0   0   0   1   2   1  98]]
     
XGBOOST antenna features importances :

    [0.04330156, 0.12807314, 0.00510513, 0.00270002, 0.00649124, 0.05210192, 0.00613526, 0.00243871, 0.00565583, 0.00450801, 0.00306396, 0.00277598, 0.02206511, 0.05703443, 0.00796382, 0.07440744, 0.00752969, 0.06535211, 0.00998182, 0.06929886, 0.01417172, 0.07100398, 0.00960822, 0.05689183, 0.01052365, 0.07067223, 0.00828135, 0.06771853, 0.0079359, , 0.04848456, 0.00806365, 0.04588205, 0.00477827]
    
##### Residents       
Residents XGBOOST 10CV f1_weighted scores : 

    [0.96872684, 0.96667148, 0.96892066, 0.96934734, 0.96814007, 0.96250368, 0.96228457, 0.96391869, 0.96409007, 0.96442005]
    
Residents XGBOOST classification report :

             precision    recall  f1-score   support
          1       0.96      0.97      0.96      2061
          2       0.97      0.98      0.97      5130
          3       0.97      0.97      0.97      6290
          4       0.97      0.97      0.97      6501
          5       0.96      0.96      0.96      4230
          6       0.96      0.97      0.97      4074
          7       0.97      0.94      0.95      4033
          8       0.96      0.96      0.96      4155
          9       0.96      0.98      0.97      3284
         10       0.96      0.97      0.97      1751
    avg / total   0.97      0.97      0.97     41509

Residents XGBOOST confusion matrix :

    [[1993   15   20   13    4    4    2    4    4    2]
     [  20 5002   43   13   22    9    9    7    3    2]
     [  20   44 6117   42   21   18    8   12    7    1]
     [  17   27   42 6284   28   33   21   30   17    2]
     [   5   26   29   33 4054   31   21   21    8    2]
     [   8   15   16   26   18 3936   14   21   14    6]
     [   7   22   30   44   32   21 3786   42   34   15]
     [   3    7   21   27   11   14   27 4008   22   15]
     [   3    1    5    7    8   13    6   18 3205   18]
     [   0    1    0    7    4    4    5   12   16 1702]]     

XGBOOST residents features importances :

    [1.37012859e-04, 9.29936650e-05, 1.47561645e-04, 1.85320372e-04, 1.88561506e-04, 1.66560902e-04, 6.71887756e-05, 1.09087414e-04, 2.37818749e-04, 1.33185793e-04, 7.94579682e-05, 1.59574323e-04, 4.52897475e-05, 3.98012417e-05, 2.11115723e-04, 8.65375623e-05, 3.22740350e-04, 3.37643251e-05, 1.25390710e-04, 2.54529994e-04, 7.22995901e-05, 1.70249172e-04, 2.10797079e-04, 0.00000000e+00, 3.13379540e-04, 2.08585363e-04, 2.17352339e-04, 8.04648444e-05, 6.23141677e-05, 1.92052481e-04, 9.85284423e-05, 1.57218674e-04, 1.30868400e-04, 7.88160105e-05, 1.36218834e-04, 7.13628688e-05, 6.75217481e-04, 8.20389483e-04, 4.59591713e-04, 2.18533547e-04, 3.73666378e-04, 8.92930781e-04, 2.80000619e-04, 4.67531616e-04, 2.66228977e-04, 1.34617920e-04, 6.64259351e-05, 1.89145998e-04, 1.48666339e-04, 1.67373946e-04, 1.80752570e-04, 1.27112289e-04, 1.97716334e-04, 7.56545342e-05, 1.91485189e-04, 2.98112980e-04, 2.83633068e-04, 2.33283339e-04, 2.10620550e-04, 1.47630097e-04, 5.54742801e-05, 6.85028717e-05, 4.82372561e-04, 1.54069887e-04, 9.55838041e-05, 6.53529132e-05, 1.18793585e-04, 2.64627015e-04, 2.46985117e-04, 5.58570937e-05, 1.38011717e-04, 8.01551942e-05, 6.38561696e-02, 1.09744258e-01, 1.22673534e-01, 1.35559663e-01, 1.07100837e-01, 9.45502222e-02, 8.89494792e-02, 9.85353515e-02, 9.83233154e-02, 5.76156974e-02, 2.35540373e-03, 6.77996548e-03]



```python
fig, ax = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('LSOA dataset feature importance', fontsize=8)

XGBOOST_clf_ant_feature_importances = [0.04330156,0.12807314,0.00510513,0.00270002,0.00649124,0.05210192\
                                       ,0.00613526,0.00243871,0.00565583,0.00450801,0.00306396,0.00277598\
                                       ,0.02206511,0.05703443,0.00796382,0.07440744,0.00752969,0.06535211\
                                       ,0.00998182,0.06929886,0.01417172,0.07100398,0.00960822,0.05689183\
                                       ,0.01052365,0.07067223,0.00828135,0.06771853,0.0079359,0.04848456\
                                       ,0.00806365,0.04588205,0.00477827]
XGBOOST_clf_res_feature_importances = [1.37012859e-04,9.29936650e-05,1.47561645e-04,1.85320372e-04\
                                       ,1.88561506e-04,1.66560902e-04,6.71887756e-05,1.09087414e-04\
                                       ,2.37818749e-04,1.33185793e-04,7.94579682e-05,1.59574323e-04\
                                       ,4.52897475e-05,3.98012417e-05,2.11115723e-04,8.65375623e-05\
                                       ,3.22740350e-04,3.37643251e-05,1.25390710e-04,2.54529994e-04\
                                       ,7.22995901e-05,1.70249172e-04,2.10797079e-04,0.00000000e+00\
                                       ,3.13379540e-04,2.08585363e-04,2.17352339e-04,8.04648444e-05\
                                       ,6.23141677e-05,1.92052481e-04,9.85284423e-05,1.57218674e-04\
                                       ,1.30868400e-04,7.88160105e-05,1.36218834e-04,7.13628688e-05\
                                       ,6.75217481e-04,8.20389483e-04,4.59591713e-04,2.18533547e-04\
                                       ,3.73666378e-04,8.92930781e-04,2.80000619e-04,4.67531616e-04\
                                       ,2.66228977e-04,1.34617920e-04,6.64259351e-05,1.89145998e-04\
                                       ,1.48666339e-04,1.67373946e-04,1.80752570e-04,1.27112289e-04\
                                       ,1.97716334e-04,7.56545342e-05,1.91485189e-04,2.98112980e-04\
                                       ,2.83633068e-04,2.33283339e-04,2.10620550e-04,1.47630097e-04\
                                       ,5.54742801e-05,6.85028717e-05,4.82372561e-04,1.54069887e-04\
                                       ,9.55838041e-05,6.53529132e-05,1.18793585e-04,2.64627015e-04\
                                       ,2.46985117e-04,5.58570937e-05,1.38011717e-04,8.01551942e-05\
                                       ,6.38561696e-02,1.09744258e-01,1.22673534e-01,1.35559663e-01\
                                       ,1.07100837e-01,9.45502222e-02,8.89494792e-02,9.85353515e-02\
                                       ,9.83233154e-02,5.76156974e-02,2.35540373e-03,6.77996548e-03]


ax[0].set_title("XGBOOST feature importance - Antenna dataset")
ax[0].bar(range(len(XGBOOST_clf_ant_feature_importances)), XGBOOST_clf_ant_feature_importances)
ax[0].set_xticks(np.arange(0, len(XGBOOST_clf_ant_feature_importances), step=1))
ax[0].set_xticklabels(antenna_dt_pd_ml_scaled.columns[:-1].values,rotation='vertical')
ax[0].set_xlim([-0.5,len(XGBOOST_clf_ant_feature_importances)-0.5])

ax[1].set_title("XGBOOST feature importance - Residents dataset")
ax[1].bar(range(len(XGBOOST_clf_res_feature_importances)), XGBOOST_clf_res_feature_importances)
ax[1].set_xticks(np.arange(0, len(XGBOOST_clf_res_feature_importances), step=1))
ax[1].set_xticklabels(residents_dt_pd_ml_scaled.columns[:-1].values,rotation='vertical')
ax[1].set_xlim([-0.5,len(XGBOOST_clf_res_feature_importances)-0.5])
plt.subplots_adjust(hspace=0.9)
```


![png](output_14_0.png)


## 4.2 Feature subsets
Computations take couple of hours - especially for residents dataset. They were run in background on the server.

Scripts:

    /home/sop/LondonMobility/ml_lsoa_analysis/antenna_subsets_script.py
    /home/sop/LondonMobility/ml_lsoa_analysis/residents_subsets_script.py

Results:

    /home/sop/LondonMobility/ml_lsoa_analysis/antenna_subsets_results.txt
    /home/sop/LondonMobility/ml_lsoa_analysis/residents_subsets_results-temp1.txt


```python
antenna_feature_subset_mobility = ['1_imd_avgctime', '1_imd_avgdevcnt', '2_imd_avgctime', '2_imd_avgdevcnt',\
                                  '3_imd_avgctime', '3_imd_avgdevcnt', '4_imd_avgctime', '4_imd_avgdevcnt',\
                                  '5_imd_avgctime', '5_imd_avgdevcnt', '6_imd_avgctime', '6_imd_avgdevcnt',\
                                  '7_imd_avgctime', '7_imd_avgdevcnt', '8_imd_avgctime', '8_imd_avgdevcnt',\
                                  '9_imd_avgctime', '9_imd_avgdevcnt', '10_imd_avgctime', '10_imd_avgdevcnt']
antenna_feature_subset_sector = ['weighted_pktloss_perc50', 'weighted_pktloss_perc10', 'weighted_pktloss_perc90',\
                                 'weighted_pktloss', 'weighted_pktretransrate_perc50', 'weighted_pktretransrate_perc10',\
                                 'weighted_pktretransrate_perc90', 'weighted_pktretransrate', 'weighted_latency_perc50',\
                                 'weighted_latency_perc10', 'weighted_latency_perc90', 'weighted_latency', 'weighted_total_volume']
residents_feature_subset_mobility34g = ['1_imd_avgtime', '2_imd_avgtime', '3_imd_avgtime', '4_imd_avgtime',\
                                      '5_imd_avgtime', '6_imd_avgtime', '7_imd_avgtime', '8_imd_avgtime',\
                                      '9_imd_avgtime', '10_imd_avgtime','3G_avgtime', '4G_avgtime']
residents_feature_subset_mobility = ['1_imd_avgtime', '2_imd_avgtime', '3_imd_avgtime', '4_imd_avgtime', '5_imd_avgtime',
                                   '6_imd_avgtime', '7_imd_avgtime', '8_imd_avgtime', '9_imd_avgtime', '10_imd_avgtime']
residents_feature_subset_perf34g = ['rtt3g_p10', 'rtt3g_p25', 'rtt3g_p50', 'rtt3g_p75',\
                                  'rtt3g_p90', 'rtt3g_min', 'rtt3g_max', 'rtt3g_avg',\
                                  'rtt3g_stdev', 'thput3g_p10', 'thput3g_p25', 'thput3g_p50',\
                                  'thput3g_p75', 'thput3g_p90', 'thput3g_min', 'thput3g_max',\
                                  'thput3g_avg', 'thput3g_stdev', 'retx3g_p10', 'retx3g_p25',\
                                  'retx3g_p50', 'retx3g_p75', 'retx3g_p90', 'retx3g_min',\
                                  'retx3g_max', 'retx3g_avg', 'retx3g_stdev', 'bytes3g_p10',\
                                  'bytes3g_p25', 'bytes3g_p50', 'bytes3g_p75', 'bytes3g_p90',\
                                  'bytes3g_min', 'bytes3g_max', 'bytes3g_avg', 'bytes3g_stdev',\
                                  'rtt4g_p10', 'rtt4g_p25', 'rtt4g_p50', 'rtt4g_p75',\
                                  'rtt4g_p90', 'rtt4g_min', 'rtt4g_max', 'rtt4g_avg',\
                                  'rtt4g_stdev', 'thput4g_p10', 'thput4g_p25', 'thput4g_p50',\
                                  'thput4g_p75', 'thput4g_p90', 'thput4g_min', 'thput4g_max',\
                                  'thput4g_avg', 'thput4g_stdev', 'retx4g_p10', 'retx4g_p25',\
                                  'retx4g_p50', 'retx4g_p75', 'retx4g_p90', 'retx4g_min',\
                                  'retx4g_max', 'retx4g_avg', 'retx4g_stdev', 'bytes4g_p10',\
                                  'bytes4g_p25', 'bytes4g_p50', 'bytes4g_p75', 'bytes4g_p90',\
                                  'bytes4g_min', 'bytes4g_max', 'bytes4g_avg', 'bytes4g_stdev']
residents_feature_subset_perf3g = ['rtt3g_p10', 'rtt3g_p25', 'rtt3g_p50', 'rtt3g_p75',\
                                  'rtt3g_p90', 'rtt3g_min', 'rtt3g_max', 'rtt3g_avg',\
                                  'rtt3g_stdev', 'thput3g_p10', 'thput3g_p25', 'thput3g_p50',\
                                  'thput3g_p75', 'thput3g_p90', 'thput3g_min', 'thput3g_max',\
                                  'thput3g_avg', 'thput3g_stdev', 'retx3g_p10', 'retx3g_p25',\
                                  'retx3g_p50', 'retx3g_p75', 'retx3g_p90', 'retx3g_min',\
                                  'retx3g_max', 'retx3g_avg', 'retx3g_stdev', 'bytes3g_p10',\
                                  'bytes3g_p25', 'bytes3g_p50', 'bytes3g_p75', 'bytes3g_p90',\
                                  'bytes3g_min', 'bytes3g_max', 'bytes3g_avg', 'bytes3g_stdev']
residents_feature_subset_perf4g = ['rtt4g_p10', 'rtt4g_p25', 'rtt4g_p50', 'rtt4g_p75',\
                                  'rtt4g_p90', 'rtt4g_min', 'rtt4g_max', 'rtt4g_avg',\
                                  'rtt4g_stdev', 'thput4g_p10', 'thput4g_p25', 'thput4g_p50',\
                                  'thput4g_p75', 'thput4g_p90', 'thput4g_min', 'thput4g_max',\
                                  'thput4g_avg', 'thput4g_stdev', 'retx4g_p10', 'retx4g_p25',\
                                  'retx4g_p50', 'retx4g_p75', 'retx4g_p90', 'retx4g_min',\
                                  'retx4g_max', 'retx4g_avg', 'retx4g_stdev', 'bytes4g_p10',\
                                  'bytes4g_p25', 'bytes4g_p50', 'bytes4g_p75', 'bytes4g_p90',\
                                  'bytes4g_min', 'bytes4g_max', 'bytes4g_avg', 'bytes4g_stdev']

```

### 4.2.1 Antenna feature subsets
#### 4.2.1.1 antenna_feature_subset_mobility
##### Logistic Regression
Antenna LR 10CV f1_weighted scores : 

    [0.87464233 0.87099324 0.87559499 0.87789664 0.87793183 0.87353143 0.88834426 0.87871298 0.87877473 0.89009563]
    
Antenna LR classification report :

             precision    recall  f1-score   support
          1       0.99      0.89      0.93       167
          2       0.95      0.89      0.92       464
          3       0.72      0.95      0.82       601
          4       0.88      0.89      0.89       511
          5       0.95      0.87      0.91       374
          6       0.95      0.86      0.90       277
          7       0.92      0.86      0.89       374
          8       0.92      0.87      0.89       313
          9       0.93      0.77      0.84       222
         10       0.95      0.84      0.90       122
    avg / total   0.89      0.88      0.88      3425

Antenna LR confusion matrix :

    [[148   5  13   1   0   0   0   0   0   0]
    [  0 415  37   5   2   2   1   2   0   0]
    [  1   6 572  12   4   1   4   1   0   0]
    [  1   2  44 457   1   3   1   1   1   0]
    [  0   2  25  14 324   0   4   2   2   1]
    [  0   1  17  11   2 239   4   1   2   0]
    [  0   1  23  15   1   4 320   4   5   1]
    [  0   1  29   3   3   1   2 271   2   1]
    [  0   4  24   3   1   2   8   7 171   2]
    [  0   0   7   0   2   0   3   6   1 103]]

#####  SVM

Antenna SVM 10CV f1_weighted scores : 

    [0.89229734 0.89079393 0.89092511 0.89640107 0.90380349 0.88742666 0.90636623 0.89424608 0.89460942 0.90848969]
    
Antenna SVM classification report :

              precision    recall  f1-score   support
           1       0.98      0.87      0.92       211
           2       0.92      0.89      0.91       472
           3       0.73      0.96      0.83       608
           4       0.89      0.91      0.90       485
           5       0.94      0.88      0.91       377
           6       0.97      0.87      0.91       290
           7       0.96      0.88      0.92       373
           8       0.94      0.88      0.91       275
           9       0.97      0.82      0.89       213
          10       0.97      0.87      0.92       121
       micro avg   0.89      0.89      0.89      3425
       macro avg   0.93      0.88      0.90      3425
    weighted avg   0.90      0.89      0.89      3425

Antenna SVM confusion matrix :

    [[183   5  18   3   1   0   1   0   0   0]
     [  2 421  43   5   0   0   1   0   0   0]
     [  1   8 581   9   4   3   2   0   0   0]
     [  0   7  31 440   4   1   1   0   1   0]
     [  0   3  24  15 330   2   1   2   0   0]
     [  0   2  20   9   2 251   2   3   0   1]
     [  0   6  24   5   4   1 328   3   0   2]
     [  0   1  24   3   3   0   1 242   1   0]
     [  0   3  21   4   2   1   2   6 174   0]
     [  0   0   9   0   0   1   1   2   3 105]]

##### XGBOOST
Antenna XGBOOST 10CV f1_weighted scores : 

    [0.91054506 0.89449145 0.90626782 0.90757051 0.91049699 0.90820279 0.91614706 0.91052841 0.90345458 0.9162234 ]
    
Antenna XGBOOST classification report :

             precision    recall  f1-score   support
          1       0.95      0.92      0.93       167
          2       0.91      0.94      0.92       464
          3       0.88      0.92      0.90       601
          4       0.89      0.91      0.90       511
          5       0.94      0.90      0.92       374
          6       0.91      0.89      0.90       277
          7       0.92      0.89      0.91       374
          8       0.88      0.92      0.90       313
          9       0.93      0.86      0.89       222
         10       0.97      0.91      0.94       122
    avg / total   0.91      0.91      0.91      3425

Antenna XGBOOST confusion matrix :

    [[153   2   9   1   1   0   1   0   0   0]
     [  1 437  14   7   3   2   0   0   0   0]
     [  2  13 552  11   5   4   7   7   0   0]
     [  4   6  22 464   3   4   3   4   1   0]
     [  0   3   9   9 336   4   7   4   2   0]
     [  0   6   7   6   3 246   4   3   1   1]
     [  0   1   8  13   2   5 334   7   4   0]
     [  0  10   1   5   3   2   1 289   1   1]
     [  1   3   2   3   1   2   7  10 191   2]
     [  0   0   0   0   1   2   0   3   5 111]]
     
Antenna XGBOOST features importances :

    [0.07272629 0.00699519 0.09784676 0.00959284 0.08834816 0.01319227 0.09198646 0.04188019 0.09016985 0.02762164 0.07793795 0.03649079 0.08911076 0.01733383 0.08712126 0.01468554 0.0578226 0.00952459 0.06327463 0.00633844]

weight : 

    387, 118, 614, 171, 704, 180, 704, 142, 551, 172, 533, 121, 519, 189, 564, 132, 545, 190, 366, 81
gain : 

    91.86994638950128, 8.836531642349152, 123.60286160857497, 12.117953089231582, 111.60394292200426, 16.664862123172224, 116.1999594263165, 52.904269995838014, 113.90515978709821, 34.892450403104654, 98.4534736203565, 46.096218819132226, 112.56729376231988, 21.896594868613757, 110.05409949804782, 18.55119477126514, 73.04318563954679, 12.031734136789474, 79.93033777566934, 8.006895520876546
    
cover : 

    974.207316871577, 295.7154528237287, 1280.193718378013, 771.3122104315789, 1467.1962246072455, 1183.2210813638887, 1460.1853623174704, 786.4332684880278, 1321.6957551313983, 575.7224700418606, 1287.6738545575042, 544.9556228239669, 1290.6680034673816, 888.6934786338626, 1204.0691104005316, 675.0086063015152, 898.8927612400001, 331.40679898426316, 806.447386885054, 295.94299940740734
    
total_gain : 

    35553.669252736996, 1042.7107337972, 75892.15702766503, 2072.1699782586006, 78569.175817091, 2999.6751821710004, 81804.77143612682, 7512.406339408998, 62761.743042691116, 6001.501469334001, 52475.701439650016, 5577.642477115, 58422.425462644016, 4138.456430168, 62070.51211689897, 2448.7577098069987, 39808.536173553, 2286.02948599, 29254.50362589498, 648.5585371910003

total_cover : 

    377018.2316293003, 34894.42343319998, 786038.9430841, 131894.3879838, 1032906.1421235008, 212979.7946455, 1027970.4950714993, 111673.52412529995, 728254.3610774005, 99024.26484720001, 686330.1644791497, 65939.6303617, 669856.693799571, 167963.06746180003, 679094.9782658998, 89101.13603180001, 489896.5548758001, 62967.29180701, 295159.7435999298, 23971.382951999996

#### 4.2.1.2 antenna_feature_subset_sector
##### Logistic regression
Antenna LR 10CV f1_weighted scores : 

    [0.13805416, 0.13828046, 0.14057277, 0.14336456, 0.13850934, 0.13203759, 0.13453722, 0.14723532, 0.15361352, 0.12636711]

Antenna LR classification report :

             precision    recall  f1-score   support
          1       0.00      0.00      0.00       165
          2       0.25      0.10      0.14       451
          3       0.23      0.68      0.34       629
          4       0.15      0.15      0.15       470
          5       0.21      0.02      0.03       381
          6       0.50      0.00      0.01       326
          7       0.20      0.22      0.21       394
          8       0.19      0.24      0.21       285
          9       0.13      0.02      0.03       207
         10       0.00      0.00      0.00       117
    avg / total   0.21      0.21      0.15      3425

Antenna LR confusion matrix :

    [[  0  22 112  13   1   0  15   0   2   0]
     [  0  46 312  36   3   0  24  27   2   1]
     [  0  42 427  71   2   0  36  47   2   2]
     [  0  27 287  72   2   0  50  28   3   1]
     [  0  15 216  64   6   0  33  42   4   1]
     [  0  14 166  50   5   1  62  27   1   0]
     [  0   9 168  63   5   0  88  54   6   1]
     [  0   9  97  50   2   0  53  69   4   1]
     [  0   1  70  40   1   1  51  39   4   0]
     [  0   0  31  16   2   0  35  30   3   0]]

##### SVM

Antenna SVM 10CV f1_weighted scores : 

    [0.11298313 0.12075874 0.13142492 0.13657083 0.12162944 0.11348738 0.12438831 0.13086537 0.12904512 0.11799336]

Antenna SVM classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00       160
           2       0.14      0.03      0.04       450
           3       0.21      0.76      0.33       600
           4       0.15      0.11      0.13       525
           5       0.25      0.01      0.02       403
           6       0.00      0.00      0.00       295
           7       0.18      0.25      0.21       373
           8       0.19      0.13      0.15       308
           9       0.18      0.01      0.02       213
          10       0.00      0.00      0.00        98
       micro avg   0.19      0.19      0.19      3425
       macro avg   0.13      0.13      0.09      3425
    weighted avg   0.16      0.19      0.12      3425

Antenna SVM confusion matrix :

    [[  0   2 135   5   1   0  15   2   0   0]
     [  0  12 364  28   4   2  24  15   1   0]
     [  0  21 458  51   3   0  36  31   0   0]
     [  0  17 367  56   1   0  59  25   0   0]
     [  0  15 238  57   5   0  57  28   3   0]
     [  0   3 176  36   2   0  56  20   2   0]
     [  0   7 202  50   0   0  95  17   2   0]
     [  0   7 148  35   1   0  77  39   1   0]
     [  0   3  84  33   2   1  69  19   2   0]
     [  0   1  44  11   1   0  35   6   0   0]]

##### XGBOOST
Antenna XGBOOST 10CV f1_weighted scores : 

    [0.16240793, 0.17564478, 0.17247159, 0.18381259, 0.17655051, 0.17447292, 0.15680946, 0.17440318, 0.1713481, 0.16904486]

Antenna XGBOOST classification report :

             precision    recall  f1-score   support
          1       0.50      0.01      0.02       169
          2       0.27      0.32      0.29       454
          3       0.23      0.57      0.33       584
          4       0.19      0.21      0.20       521
          5       0.26      0.04      0.07       374
          6       0.43      0.02      0.04       324
          7       0.24      0.25      0.24       392
          8       0.22      0.28      0.25       284
          9       0.27      0.04      0.07       207
         10       0.00      0.00      0.00       116
    avg / total   0.26      0.23      0.19      3425

Antenna XGBOOST confusion matrix :

    [[  2  44  96  12   2   0   5   8   0   0]
     [  0 146 201  63   4   2  17  20   1   0]
     [  1 104 333  63   8   2  36  35   2   0]
     [  0  76 245 107   3   0  48  39   3   0]
     [  1  53 151  73  16   2  40  36   2   0]
     [  0  43 132  54   8   6  52  25   2   2]
     [  0  40 118  74  11   0  97  47   5   0]
     [  0  21  78  55   3   1  43  79   3   1]
     [  0  12  60  36   3   1  48  39   8   0]
     [  0   7  22  27   4   0  26  26   4   0]]

Antenna XGBOOST features importances :

    [0.07677774, 0.06178833, 0.11400992, 0.11198197, 0.07180817, 0.05961196, 0.06530163, 0.07936504, 0.06343717, 0.06280398, 0.05802283, 0.10859713, 0.06649413]
    
weight : 

    339, 252, 1025, 730, 381, 195, 602, 718, 416, 358, 486, 652, 572

gain : 

    8.745658918418588, 7.038233427694442, 12.986731433056569, 12.75573095705205, 8.179581056868761, 6.790326190415387, 7.438429999158373, 9.040374658295272, 7.226051253016826, 7.153924655999996, 6.609309482821399, 12.370166710441719, 7.574265651096149
    
cover : 

    2548.1861665903834, 1374.9899412124994, 2898.731156220765, 2591.2924033500226, 2371.0986452607876, 1818.743179446358, 2499.5100308189726, 2716.9650045011326, 2268.4748479081254, 1984.104435835949, 1629.2515665594647, 2516.5634227797545, 2357.37202375093

total_gain : 

    2964.7783733439014, 1773.6348237789994, 13311.399718882984, 9311.683598647996, 3116.4203826669977, 1324.1136071310004, 4477.934859493341, 6490.989004656006, 3006.037321255, 2561.1050268479985, 3212.1244086512, 8065.348695208001, 4332.479952426997
    
total_cover : 
    
    863835.1104741399, 346497.46518554987, 2971199.435126284, 1891643.4544455165, 903388.5838443601, 354654.9199920398, 1504705.0385530214, 1950780.8732318133, 943685.5367297803, 710309.3880292698, 791816.2613478998, 1640799.3516524, 1348416.797585532

### 4.2.2 Residents feature subsets
#### 4.2.2.1 residents_feature_subset_mobility34g

##### Logistic Regression
Residents LR 10CV f1_weighted scores : 

    [0.96801052, 0.96564504, 0.96829921, 0.96764524, 0.96648558, 0.96125696, 0.96072303, 0.96327722, 0.96209204, 0.96322882]

Residents LR classification report :

             precision    recall  f1-score   support
          1       0.97      0.97      0.97      2080
          2       0.97      0.97      0.97      5300
          3       0.97      0.97      0.97      6318
          4       0.96      0.96      0.96      6477
          5       0.96      0.96      0.96      4113
          6       0.96      0.96      0.96      3862
          7       0.96      0.95      0.95      4019
          8       0.97      0.96      0.96      4257
          9       0.96      0.96      0.96      3305
         10       0.96      0.97      0.97      1778
    avg / total   0.96      0.96      0.96     41509

Residents LR confusion matrix :

    [[2016   14   14   16    7    5    3    3    2    0]
     [  15 5151   52   29   19   13   14    5    2    0]
     [  18   59 6112   55   20   21   18    8    6    1]
     [   9   38   43 6246   36   27   26   33   14    5]
     [   2   21   19   30 3968   21   21   10   11   10]
     [   6   14   13   17   23 3713   21   22   27    6]
     [   9   13   18   38   29   27 3823   24   28   10]
     [   4   10   21   36   16   15   37 4079   22   17]
     [   4    3    9   17    5   12   27   16 3187   25]
     [   1    0    0    3    3   11    8   13   10 1729]]

##### SVM
Residents SVM 10CV f1_weighted scores : 

    [0.96798351 0.96602959 0.96764579 0.96747357 0.96631473 0.96103876 0.96108149 0.96339524 0.96223355 0.96290974]

Residents SVM classification report :

              precision    recall  f1-score   support

           1       0.96      0.97      0.97      2130
           2       0.97      0.97      0.97      5249
           3       0.97      0.97      0.97      6167
           4       0.97      0.96      0.97      6433
           5       0.96      0.96      0.96      4334
           6       0.96      0.96      0.96      4050
           7       0.96      0.95      0.96      3990
           8       0.96      0.97      0.96      4148
           9       0.97      0.97      0.97      3216
          10       0.96      0.97      0.97      1792
        accuracy                       0.96     41509
       macro avg   0.96      0.97      0.96     41509
    weighted avg   0.96      0.96      0.96     41509

Residents SVM confusion matrix :

    [[2064   18   17   13    7    5    2    1    2    1]
     [  17 5097   60   28   15    8   11    8    2    3]
     [  28   47 5970   39   28   21   12   15    7    0]
     [  15   51   50 6188   27   30   22   31   10    9]
     [  12   27   30   30 4169   18   21   16    5    6]
     [   3   13   23   25   22 3892   19   26   17   10]
     [   6   17   15   28   31   33 3782   39   24   15]
     [   1    9   12   22   12   22   26 4005   20   19]
     [   1    1    4    7   11   15   23   14 3122   18]
     [   0    1    2    3    1    3    6    8   21 1747]]

     
##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.96887135, 0.96708232, 0.96904094, 0.9692261, 0.9683074, 0.96259957, 0.96221344, 0.96399242, 0.96425773, 0.96480753]

Residents XGBOOST classification report :

             precision    recall  f1-score   support
          1       0.97      0.97      0.97      2080
          2       0.97      0.97      0.97      5300
          3       0.97      0.97      0.97      6318
          4       0.96      0.97      0.97      6477
          5       0.97      0.96      0.96      4113
          6       0.96      0.96      0.96      3862
          7       0.97      0.95      0.96      4019
          8       0.97      0.96      0.97      4257
          9       0.96      0.97      0.96      3305
         10       0.96      0.98      0.97      1778
    avg / total   0.97      0.97      0.97     41509

Residents XGBOOST confusion matrix :

    [[2025   14   12   14    5    2    2    3    3    0]
     [  16 5162   49   27   15   12    9    5    4    1]
     [  22   62 6115   48   16   21   14   12    6    2]
     [   9   38   40 6280   27   31   16   24    7    5]
     [   2   25   22   36 3951   28   15   13   11   10]
     [   8   18   15   20   21 3714   18   18   27    3]
     [   9   15   17   41   25   25 3818   25   32   12]
     [   2    8   20   33   14   16   25 4094   27   18]
     [   3    2    8   18    7   12   23   14 3193   25]
     [   1    0    0    2    3    8    4   12   10 1738]]
 
Residents XGBOOST features importances :

    [0.06933898, 0.10777035, 0.13145292, 0.13331982, 0.10040218, 0.09977605, 0.08847483, 0.10575587, 0.09498658, 0.05998726, 0.00254729, 0.00618791]

weight : 
    
    409, 503, 452, 455, 452, 443, 488, 441, 419, 412, 820, 1164
    
gain : 

    1267.8958930602157, 1970.631436895257, 2403.678294964348, 2437.815374650628, 1835.901061774175, 1824.4518744018671, 1617.8038765187382, 1933.795537281561, 1736.874106777725, 1096.8950953926442, 46.578359463828164, 113.14883935461259
cover : 

    9514.811376169017, 10108.216077766318, 11840.887070607938, 12018.062846480923, 9913.994128833201, 10083.108636410992, 9426.196544733662, 10061.33150947546, 9882.437824797096, 9138.416127730268, 4241.107811247085, 982.4192356066596

total_gain : 
    
    518569.4202616282, 991227.6127583142, 1086462.5893238853, 1109205.9954660358, 829827.2799219271, 808232.1803600271, 789488.2917411443, 852803.8319411684, 727750.2507398668, 451920.7793017694, 38194.25476033909, 131705.24900876905
    
total_cover : 

    3891557.8528531278, 5084432.687116458, 5352080.955914788, 5468218.59514882, 4481125.346232607, 4466817.12593007, 4599983.913830027, 4437047.195678677, 4140741.4485899834, 3765027.4446248705, 3477708.4052226096, 1143535.9902461518

#### 4.2.2.2 residents_feature_subset_mobility
##### Logistic Regression
Residents LR 10CV f1_weighted scores : 

    [0.96723883, 0.9656719, , 0.96774503, 0.9670678, , 0.96550051, 0.96070231, 0.96028815, 0.96318051, 0.96134703, 0.96286912]

Residents LR classification report :

             precision    recall  f1-score   support
          1       0.96      0.96      0.96      2168
          2       0.97      0.97      0.97      5284
          3       0.97      0.97      0.97      6143
          4       0.96      0.96      0.96      6527
          5       0.97      0.96      0.96      4139
          6       0.96      0.96      0.96      3883
          7       0.95      0.95      0.95      4019
          8       0.96      0.97      0.96      4291
          9       0.96      0.97      0.96      3213
         10       0.96      0.96      0.96      1842
    avg / total   0.96      0.96      0.96     41509

Residents LR confusion matrix :

    [[2073   30   28   16    4    3    6    6    2    0]
     [  16 5129   44   31   14   18   19    9    4    0]
     [  11   49 5960   36   27   20   17   16    7    0]
     [  25   33   44 6277   32   23   31   37   16    9]
     [   5   16   31   32 3975   32   19    9   16    4]
     [   5   11   16   29   21 3722   25   25   20    9]
     [   7   14   15   37   24   27 3835   23   24   13]
     [   2    6    9   32   13   20   38 4149    8   14]
     [   3    1    8   14    5    7   22   22 3111   20]
     [   3    2    1    2    2   12   10   12   27 1771]]

##### SVM

Residents SVM 10CV f1_weighted scores : 

    [0.9670659  0.96564674 0.96750235 0.96737808 0.96576134 0.96087 0.96038345 0.96298649 0.9616319  0.96254934]

Residents SVM classification report :

              precision    recall  f1-score   support
           1       0.97      0.96      0.97      2129
           2       0.97      0.97      0.97      5250
           3       0.97      0.97      0.97      6281
           4       0.96      0.96      0.96      6457
           5       0.96      0.96      0.96      4171
           6       0.96      0.96      0.96      3901
           7       0.95      0.95      0.95      3967
           8       0.96      0.96      0.96      4294
           9       0.97      0.97      0.97      3257
          10       0.96      0.97      0.97      1802
        accuracy                       0.96     41509
       macro avg   0.96      0.96      0.96     41509
    weighted avg   0.96      0.96      0.96     41509

Residents SVM confusion matrix :

    [[2054   17   15   18    7    4    5    5    4    0]
     [  22 5102   35   36   24    6   16    3    5    1]
     [  15   47 6094   41   22   23   18   16    5    0]
     [  14   28   42 6205   33   26   39   47   16    7]
     [   5   29   26   29 4006   20   26   20    8    2]
     [   3   15   20   36   18 3732   35   23   11    8]
     [   2   16   15   37   19   23 3787   30   25   13]
     [   2    7   10   32   25   21   32 4135   18   12]
     [   4    1    2    8   13   22   19   15 3150   23]
     [   0    0    1    2    1    5   10   22   18 1743]]
     
##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.96742491 0.96603107 0.96767239 0.96751822 0.96641158 0.96135344 0.96093684 0.96373318 0.96199578 0.96346574]

Residents XGBOOST classification report :

              precision    recall  f1-score   support
           1       0.96      0.96      0.96      2128
           2       0.97      0.97      0.97      5165
           3       0.97      0.97      0.97      6251
           4       0.96      0.96      0.96      6581
           5       0.96      0.95      0.96      4226
           6       0.96      0.96      0.96      3979
           7       0.96      0.95      0.95      3950
           8       0.96      0.97      0.96      4193
           9       0.97      0.97      0.97      3305
          10       0.96      0.98      0.97      1731
       micro avg   0.96      0.96      0.96     41509
       macro avg   0.96      0.96      0.96     41509
    weighted avg   0.96      0.96      0.96     41509

Residents XGBOOST confusion matrix :

    [[2050   15   23    9    9    5   11    5    1    0]
     [  22 5033   37   29   18   12   11    2    1    0]
     [  24   54 6047   50   19   12   20   21    4    0]
     [  20   37   41 6347   24   31   30   32   12    7]
     [   4   31   30   37 4033   28   24   20    9   10]
     [   5   14   19   32   26 3821   21   19   17    5]
     [   7   19   17   38   27   17 3745   43   24   13]
     [   1    4    8   33   12   28   17 4053   18   19]
     [   2    3    7    9    9   18   21   17 3198   21]
     [   1    0    0    2    3    8    7   11   11 1688]]
 
Residents XGBOOST features importances :

    [0.07534941 0.11860289 0.11598326 0.12136484 0.11365965 0.09466233 0.09071651 0.10421316 0.09519716 0.07025084]

weight : 
    
    552 , 670 , 749 , 730 , 579 , 682 , 695 , 654 , 611 , 517 ,
    
gain : 

    959.068115693094 , 1509.6103925447424 , 1476.2669414508293 , 1544.7652498494429 , 1446.6913591524112 , 1204.8882670251849 , 1154.6647499216 , 1326.4538177878119 , 1211.695650024539 , 894.1721461235913 ,
cover : 

    7828.899838660745 , 8948.193235478726 , 8205.29285387698 , 8728.183624686499 , 8436.81955822828 , 7619.121360153812 , 7484.380951259562 , 7644.973229158556 , 7826.4065922733425 , 7999.85896735222 ,

total_gain : 

    529405.5998625879 , 1011438.9630049773 , 1105723.9391466712 , 1127678.6323900933 , 837634.2969492461 , 821733.798111176 , 802492.0011955121 , 867500.7968332289 , 740346.0421649932 , 462286.9995458967 ,

total_cover : 

    4321552.710940732 , 5995289.467770746 , 6145764.347553858 , 6371574.046021144 , 4884918.524214175 , 5196240.7676249 , 5201644.761125395 , 4999812.4918696955 , 4781934.427879012 , 4135927.0861210977 ,

#### 4.2.2.3 residents_feature_subset_perf34g

##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.07677863 0.0780163  0.07893931 0.07902017 0.07808913 0.07844307 0.07812221 0.07913468 0.07897273 0.07920725]
    
Residents LR classification report :

              precision    recall  f1-score   support
           1       0.40      0.00      0.00      2066
           2       0.15      0.02      0.03      5216
           3       0.17      0.30      0.21      6104
           4       0.16      0.72      0.26      6605
           5       0.14      0.00      0.00      4219
           6       0.00      0.00      0.00      3834
           7       0.17      0.00      0.00      4051
           8       0.13      0.00      0.01      4231
           9       0.15      0.00      0.00      3314
          10       0.00      0.00      0.00      1869
        accuracy                       0.16     41509
       macro avg   0.15      0.10      0.05     41509
    weighted avg   0.15      0.16      0.08     41509

Residents LR confusion matrix :

    [[   2   46  671 1333    2    0    3    8    1    0]
     [   0  101 1696 3404    1    0    4    9    1    0]
     [   1  137 1801 4146    2    0    6   10    1    0]
     [   1  117 1702 4756    2    0    3   20    3    1]
     [   0   73 1143 2989    2    0    7    4    1    0]
     [   1   48  931 2839    0    0    2   13    0    0]
     [   0   53  957 3024    3    0    6    8    0    0]
     [   0   42  898 3270    1    0    3   14    3    0]
     [   0   31  623 2645    0    0    2   11    2    0]
     [   0   16  372 1472    1    0    0    7    1    0]]

##### SVM

Residents SVM 10CV f1_weighted scores : 

    [0.07941807 0.08322136 0.08565015 0.08299262 0.07918682 0.07800307 0.08080449 0.08963281 0.09175859 0.07810846]

Residents SVM classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2107
           2       0.15      0.02      0.03      5101
           3       0.17      0.30      0.22      6177
           4       0.16      0.72      0.26      6554
           5       0.20      0.00      0.00      4236
           6       0.00      0.00      0.00      4003
           7       0.14      0.00      0.00      4052
           8       0.13      0.00      0.01      4248
           9       0.04      0.00      0.00      3251
          10       0.00      0.00      0.00      1780
       micro avg   0.16      0.16      0.16     41509
       macro avg   0.10      0.10      0.05     41509
    weighted avg   0.12      0.16      0.08     41509

Residents SVM confusion matrix :

    [[   0   42  695 1363    0    0    1    6    0    0]
     [   0   87 1641 3347    1    0    3   18    4    0]
     [   0  109 1857 4176    2    0    8   21    4    0]
     [   0   92 1732 4704    0    0    1   20    5    0]
     [   0   57 1203 2966    1    0    3    6    0    0]
     [   0   45  980 2967    0    0    3    7    1    0]
     [   0   51  966 3017    0    1    4   11    2    0]
     [   0   46  907 3268    1    0    4   16    6    0]
     [   0   27  647 2562    0    0    1   13    1    0]
     [   0   12  367 1396    0    0    0    3    2    0]]

##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.08454878 0.08684764 0.08665237 0.08478338 0.08531096 0.08243307 0.08486658 0.08588638 0.08438759 0.08702684]

Residents XGBOOST classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2045
           2       0.20      0.06      0.09      5158
           3       0.18      0.34      0.23      6225
           4       0.16      0.68      0.26      6521
           5       0.33      0.00      0.00      4213
           6       0.25      0.00      0.00      3979
           7       0.41      0.00      0.00      3991
           8       0.25      0.00      0.00      4232
           9       0.00      0.00      0.00      3331
          10       0.00      0.00      0.00      1814
        accuracy                       0.16     41509
       macro avg   0.18      0.11      0.06     41509
    weighted avg   0.20      0.16      0.09     41509

Residents XGBOOST confusion matrix :

    [[   0   89  730 1226    0    0    0    0    0    0]
     [   0  285 1746 3123    0    1    3    0    0    0]
     [   0  245 2095 3880    3    1    1    0    0    0]
     [   0  225 1851 4443    0    0    1    1    0    0]
     [   0  162 1262 2786    2    0    0    1    0    0]
     [   0  127 1029 2822    0    1    0    0    0    0]
     [   0  108  969 2907    0    0    7    0    0    0]
     [   0  107  979 3144    0    1    0    1    0    0]
     [   0   69  708 2549    1    0    4    0    0    0]
     [   0   38  371 1403    0    0    1    1    0    0]]

Residents XGBOOST features importances :

    [0.01259051 0.01601386 0.01054224 0.01338685 0.01522961 0.01283618
     0.01637111 0.0122337  0.01182944 0.01175465 0.01260043 0.01034413
     0.01008887 0.01123273 0.01240997 0.01154933 0.01164784 0.00897067
     0.00976243 0.01111055 0.01212237 0.01915771 0.01474634 0.01068356
     0.01899059 0.01315921 0.01903136 0.01048751 0.01119131 0.01176496
     0.01112677 0.01154399 0.01214164 0.00941978 0.00933004 0.00923542
     0.02731442 0.01516874 0.0096233  0.01059497 0.02438811 0.035053
     0.01084263 0.02425995 0.02052793 0.00909843 0.00960398 0.01007483
     0.01055795 0.01069874 0.01059428 0.01074526 0.01070941 0.01079801
     0.01041955 0.01398481 0.01274    0.01701309 0.01228515 0.00795762
     0.04862444 0.01483164 0.0101317  0.01246554 0.01331067 0.01214701
     0.01289247 0.01106155 0.02679022 0.01981028 0.01251695 0.01372566]

weight : 

    71 , 83 , 64 , 84 , 117 , 50 , 124 , 49 , 104 , 59 , 64 , 46 , 34 , 21 , 90 , 29 , 32 , 58 , 23 , 29 , 96 , 92 , 87 , 23 , 162 , 59 , 84 , 51 , 54 , 53 , 72 , 80 , 119 , 73 , 56 , 55 , 218 , 120 , 86 , 78 , 195 , 353 , 141 , 86 , 175 , 53 , 39 , 46 , 36 , 29 , 90 , 68 , 29 , 53 , 33 , 65 , 85 , 115 , 105 , 18 , 496 , 71 , 77 , 109 , 113 , 98 , 177 , 127 , 211 , 414 , 89 , 135 ,

gain : 

    9.044298760639437 , 11.503441805855426 , 7.5729457992187506 , 9.616343994166671 , 10.94008007390599 , 9.220774976 , 11.760071981612906 , 8.787987299591839 , 8.497594973461538 , 8.443865308135598 , 9.051426532812503 , 7.430628205652172 , 7.247266229411764 , 8.068950405238095 , 8.914609503444442 , 8.296377762413792 , 8.3671402234375 , 6.444017945172414 , 7.012770871304345 , 7.981183815948276 , 8.708016133645833 , 13.761804695978256 , 10.59292469114942 , 7.674458336521739 , 13.641751233765422 , 9.452822379152536 , 13.671042307023816 , 7.533629473529412 , 8.039199203703703 , 8.451275032830187 , 7.992834259166664 , 8.292545597500002 , 8.721856342941177 , 6.766632050684934 , 6.702169690892859 , 6.634198566309093 , 19.621115985688075 , 10.89635308658334 , 6.91282694255814 , 7.610818236256407 , 17.51902164211794 , 25.180073734419263 , 7.788727799333332 , 17.426958346627913 , 14.746086715771414 , 6.535794298679247 , 6.898946735897437 , 7.237177662482607 , 7.584229499166666 , 7.685365745172415 , 7.61032287968889 , 7.718776845588235 , 7.693025062413793 , 7.756672660566034 , 7.484811348787879 , 10.045889382615385 , 9.151686431411767 , 12.221230195817393 , 8.824950765809527 , 5.71630052611111 , 34.92901571018956 , 10.654203442112676 , 7.2780343426753245 , 8.954527543944952 , 9.561623340530971 , 8.725717446122454 , 9.2612132077226 , 7.945985031889765 , 19.24455920142179 , 14.230569014541041 , 8.991461367078651 , 9.859730458962968 ,

cover : 

    22788.604838943666 , 17921.925508164335 , 20450.550294839217 , 25999.375301988322 , 31452.56139038462 , 26331.894607907394 , 27974.645531091126 , 26076.859720308163 , 30630.45474416692 , 28000.902834056782 , 22421.271429373744 , 17649.39919495434 , 35239.94890358558 , 12987.526800885713 , 39471.44116401023 , 18662.331176515865 , 11676.699214703125 , 21614.61627821224 , 9859.263567193044 , 20010.42243504103 , 22875.692774792176 , 32004.027157078148 , 22693.32355496425 , 28864.269402173904 , 27685.27930041036 , 17047.472460717112 , 23138.356513033927 , 33569.46200320863 , 23995.25584041612 , 31068.60096620754 , 19670.374427073613 , 20857.67597775475 , 22932.24939693051 , 18520.051716429996 , 27044.672377447325 , 25224.85470667309 , 30126.822781343195 , 22509.772921303094 , 37017.23610434895 , 25786.97432332526 , 38048.026203541005 , 36947.953516579604 , 28731.932864795464 , 21292.11280556686 , 31916.014779421716 , 22848.40452827246 , 11843.967460348207 , 20500.077401466522 , 15221.639978868887 , 23251.026172848968 , 33352.28441304388 , 24914.087783733674 , 17788.706275824137 , 20867.20896205075 , 28616.098033268183 , 39754.17799467077 , 24645.62479193377 , 35304.60230738982 , 23402.9216672822 , 52127.02762161111 , 38677.646662681276 , 26877.02945935493 , 15355.622091902598 , 30995.675758425677 , 30718.681436649735 , 32268.488613053487 , 36572.35497468119 , 26039.5064581411 , 32507.078658104354 , 33009.780305241606 , 25777.736303740337 , 26091.121979143543 ,

total_gain : 

    642.1452120054 , 954.7856698860004 , 484.66853115000004 , 807.7728955100004 , 1279.989368647001 , 461.03874879999995 , 1458.2489257200004 , 430.6113776800001 , 883.74987724 , 498.1880531800003 , 579.2912981000002 , 341.8088974599999 , 246.40705179999998 , 169.44795850999998 , 802.3148553099999 , 240.59495510999997 , 267.74848715 , 373.75304082 , 161.29373003999993 , 231.45433066249998 , 835.96954883 , 1266.0860320299996 , 921.5844481299996 , 176.51254174 , 2209.9636998699984 , 557.7165203699997 , 1148.3675537900006 , 384.21510315 , 434.11675699999995 , 447.9175767399999 , 575.4840666599998 , 663.4036478000002 , 1037.90090481 , 493.9641397000002 , 375.3215026900001 , 364.88092114700015 , 4277.403284880001 , 1307.5623703900008 , 594.50311706 , 593.6438224279998 , 3416.2092202129984 , 8888.56602825 , 1098.2106197059998 , 1498.7184178100006 , 2580.5651752599974 , 346.39709783000006 , 269.05892270000004 , 332.91017247419995 , 273.03226197 , 222.87560661000003 , 684.929059172 , 524.8768255 , 223.09772680999998 , 411.1036510099998 , 246.99877451 , 652.9828098700001 , 777.8933466700001 , 1405.4414725190002 , 926.6198304100003 , 102.89340946999998 , 17324.791792254022 , 756.44844439 , 560.408644386 , 976.0435022899998 , 1080.4634374799998 , 855.1203097200005 , 1639.2347377669003 , 1009.1400990500001 , 4060.601991499998 , 5891.455572019991 , 800.2400616699999 , 1331.0636119600006 ,

total_cover : 

    1617990.9435650003 , 1487519.8171776398 , 1308835.2188697099 , 2183947.525367019 , 3679949.6826750007 , 1316594.7303953697 , 3468856.0458552996 , 1277766.1262951 , 3185567.2933933595 , 1652053.2672093501 , 1434961.3714799196 , 811872.3629678998 , 1198158.2627219097 , 272738.0628186 , 3552429.7047609207 , 541207.6041189601 , 373654.3748705 , 1253647.74413631 , 226763.06204544002 , 580302.2506161899 , 2196066.506380049 , 2944370.4984511896 , 1974319.1492818897 , 663878.1962499998 , 4485015.246666478 , 1005800.8751823097 , 1943621.9470948498 , 1712042.5621636403 , 1295743.8153824706 , 1646635.8512089998 , 1416266.9587493 , 1668614.07822038 , 2728937.678234731 , 1351963.7752993896 , 1514501.65313705 , 1387367.00886702 , 6567647.366332817 , 2701172.750556371 , 3183482.3049740097 , 2011383.9972193702 , 7419365.109690496 , 13042627.5913526 , 4051202.5339361606 , 1831121.7012787499 , 5585302.5863988 , 1210965.4399984402 , 461914.73095358006 , 943003.5604674601 , 547979.0392392799 , 674279.7590126201 , 3001705.5971739492 , 1694157.9692938898, 515872.4819989 , 1105962.0749886897 , 944331.2350978501 , 2584021.5696536 , 2094878.1073143703 , 4060029.265349829 , 2457306.775064631 , 938286.497189 , 19184112.74468991 , 1908269.0916142 , 1182382.9010765 , 3378528.6576683987 , 3471211.00234142 , 3162311.8840792417 , 6473306.830518571 , 3307017.3201839197 , 6858993.596860019 , 13666049.046370026 , 2294218.53103289 , 3522301.4671843783 ,

#### 4.2.2.4 residents_feature_subset_perf3g

##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.0669874  0.06769562 0.06780634 0.06702892 0.06756357 0.06914617 0.07130319 0.06924304 0.06950139 0.06866478]

Residents LR classification report :

              precision    recall  f1-score   support
           1       0.67      0.00      0.00      2092
           2       0.19      0.00      0.01      5155
           3       0.18      0.19      0.18      6255
           4       0.16      0.84      0.27      6550
           5       0.00      0.00      0.00      4143
           6       0.00      0.00      0.00      4006
           7       0.31      0.00      0.00      4007
           8       0.00      0.00      0.00      4235
           9       0.00      0.00      0.00      3309
          10       0.00      0.00      0.00      1757
        accuracy                       0.16     41509
       macro avg   0.15      0.10      0.05     41509
    weighted avg   0.14      0.16      0.07     41509

Residents LR confusion matrix :

    [[   2    7  424 1659    0    0    0    0    0    0]
     [   1   20 1019 4114    0    0    0    1    0    0]
     [   0   13 1174 5065    1    0    2    0    0    0]
     [   0   19 1033 5489    2    0    3    2    2    0]
     [   0   10  627 3502    0    0    2    1    1    0]
     [   0   11  565 3429    0    0    0    1    0    0]
     [   0   10  650 3342    0    0    4    1    0    0]
     [   0    3  586 3645    0    0    0    0    1    0]
     [   0    5  390 2909    2    0    2    1    0    0]
     [   0    5  239 1513    0    0    0    0    0    0]]
 
##### SVM

Residents SVM 10CV f1_weighted scores : 

    [0.06958392 0.06955987 0.06943512 0.07363844 0.06735588 0.07110175 0.07175356 0.06891826 0.07388368 0.06894311]

Residents SVM classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2172
           2       0.16      0.00      0.01      5217
           3       0.16      0.19      0.18      6123
           4       0.16      0.81      0.26      6489
           5       0.00      0.00      0.00      4171
           6       0.00      0.00      0.00      3941
           7       0.08      0.00      0.00      4059
           8       0.07      0.00      0.00      4241
           9       0.09      0.00      0.00      3259
          10       0.00      0.00      0.00      1837
        accuracy                       0.16     41509
       macro avg   0.07      0.10      0.05     41509
    weighted avg   0.09      0.16      0.07     41509

Residents SVM confusion matrix :

    [[   0    5  459 1701    0    0    0    4    3    0]
     [   0   26 1073 4094    0    0    1   17    6    0]
     [   0   40 1184 4872    1    0    0   21    5    0]
     [   0   26 1168 5264    1    0    5   21    4    0]
     [   0   17  728 3407    0    0    2    8    9    0]
     [   0    5  669 3251    0    0    0   13    3    0]
     [   0    8  636 3396    0    0    1   13    5    0]
     [   0   14  629 3586    0    0    0    8    4    0]
     [   0   15  422 2804    0    0    2   12    4    0]
     [   0    5  281 1544    0    0    1    4    2    0]]
 
##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.07697958 0.07877342 0.07831757 0.07844113 0.07906996 0.07944121 0.07871002 0.07948972 0.07923951 0.07879694]

Residents XGBOOST classification report :

                precision    recall  f1-score   support
           1       0.00      0.00      0.00      2100
           2       0.19      0.03      0.06      5156
           3       0.17      0.29      0.22      6234
           4       0.16      0.73      0.26      6469
           5       0.25      0.00      0.00      4214
           6       0.00      0.00      0.00      3941
           7       0.60      0.00      0.00      3989
           8       0.17      0.00      0.00      4230
           9       0.00      0.00      0.00      3379
          10       0.00      0.00      0.00      1797
       micro avg   0.16      0.16      0.16     41509
       macro avg   0.15      0.11      0.05     41509
    weighted avg   0.17      0.16      0.08     41509

Residents XGBOOST confusion matrix :

    [[   0   62  624 1414    0    0    0    0    0    0]
     [   0  166 1525 3462    2    0    1    0    0    0]
     [   0  141 1826 4263    2    0    0    2    0    0]
     [   0  124 1607 4734    3    0    1    0    0    0]
     [   0   86 1096 3027    4    1    0    0    0    0]
     [   0   68  929 2940    2    0    0    2    0    0]
     [   0   70  926 2987    3    0    3    0    0    0]
     [   0   76  938 3215    0    0    0    1    0    0]
     [   0   35  667 2677    0    0    0    0    0    0]
     [   0   27  356 1413    0    0    0    1    0    0]]

Residents XGBOOST features importances :
    
    [0.04058945 0.03945416 0.03025115 0.03212867 0.03238182 0.03397333 0.02763107 0.03483811 0.02270295 0.02350652 0.02555693 0.0258921 0.02230483 0.02051507 0.0308361  0.01969866 0.01876105 0.0199656 0.0185577  0.02180434 0.02350835 0.03398243 0.03568126 0.02109945 0.05153529 0.03392423 0.04071657 0.02689304 0.0361452  0.02594105 0.0203705  0.02044205 0.02733919 0.02176205 0.01843257 0.02087701]
    
weight : 

    268 , 192 , 236 , 248 , 240 , 230 , 256 , 127 , 206 , 167 , 116 , 153 , 119 , 65 , 193 , 80 , 67 , 112 , 40 , 82 , 201 , 196 , 187 , 32 , 547 , 172 , 357 , 158 , 222 , 196 , 249 , 216 , 223 , 352 , 157 , 197 ,

gain : 

    11.432717471802242 , 11.112943406770837 , 8.520756527559321 , 9.049593352366932 , 9.120897008304171 , 9.569171816713048 , 7.782765526863278 , 9.812752480078744 , 6.394677673601945 , 6.621014517101799 , 7.198550050353449 , 7.292955049908496 , 6.28253841252101 , 5.778422095615385 , 8.6855185619171 , 5.548465124125 , 5.284373603432837 , 5.623655969348218 , 5.227095005924999 , 6.141566800926828 , 6.621530475422887 , 9.57173523887755 , 10.050241268534755 , 5.943022683125 , 14.51580272476051 , 9.555342128988375 , 11.468523528700281 , 7.574888166518991 , 10.180915864953997 , 7.306742070979591 , 5.737702773453811 , 5.757854399125 , 7.700552800493275 , 6.129655690909089 , 5.191849910757962 , 5.880367783685277 ,

cover : 

    29200.94745178928 , 25394.528201777695 , 24971.034859845524 , 26364.722791897813 , 33525.33814320852 , 24903.208122548825 , 28731.28754707634 , 28080.833983819066 , 31815.649253990407 , 38500.65001211915 , 24501.217012503716 , 22765.218469875686 , 32916.280177422195 , 16216.878969412466 , 38156.881227082675 , 35843.17876480475 , 19178.04530223105 , 22155.927152043474 , 12577.591448249997 , 21592.908424168774 , 31104.23005928159 , 26307.707774457347 , 24230.25700719299 , 43726.7781301375 , 31736.20979040337 , 25790.68059363691 , 36716.15061776613 , 25789.949997410888 , 25916.852251726174 , 34114.925621945775 , 27177.041464046124 , 29396.657128015373 , 26379.863957202757 , 33110.15799836169 , 30948.180538440378 , 24680.68596075638 ,

total_gain : 

    3063.968282443001 , 2133.6851341000006 , 2010.8985405039996 , 2244.2991513869993 , 2189.015281993001 , 2200.909517844001 , 1992.3879748769991 , 1246.2195649700006 , 1317.3036007620005 , 1105.7094243560005 , 835.0318058410002 , 1115.822122636 , 747.6220710900002 , 375.597436215 , 1676.30508245 , 443.87720993000005 , 354.05303143000003 , 629.8494685670004 , 209.083800237 , 503.6084776759999 , 1330.9276255600003 , 1876.0601068199996 , 1879.395117215999 , 190.17672586 , 7940.144090443999 , 1643.5188461860005 , 4094.2628997460006 , 1196.8323303100005 , 2260.1633220197873 , 1432.121445912 , 1428.687990589999 , 1243.696550211 , 1717.2232745100005 , 2157.638803199999 , 815.120435989 , 1158.4324533859997 ,

total_cover : 

    7825853.917079528 , 4875749.414741318 , 5893164.226923544 , 6538451.252390658 , 8046081.154370045 , 5727737.86818623 , 7355209.612051543 , 3566265.9159450214 , 6554023.746322024 , 6429608.552023898 , 2842141.173450431 , 3483078.4258909803 , 3917037.341113241 , 1054097.1330118102 , 7364278.076826956 , 2867454.30118438 , 1284929.0352494803 , 2481463.841028869 , 503103.6579299999 , 1770618.4907818395 , 6251950.241915599 , 5156310.72379364 , 4531058.060345089 , 1399256.9001644 , 17359706.755350642 , 4435997.062105549 , 13107665.770542508 , 4074812.0995909204 , 5753541.1998832105 , 6686525.421901371 , 6767083.3245474845 , 6349677.939651321 , 5882709.662456214 , 11654775.615423316 , 4858864.344535139 , 4862095.134269007 ,

#### 4.2.2.5 residents_feature_subset_perf4g
##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.07521661 0.07617969 0.0764625  0.07697479 0.07652408 0.0752466 0.07561458 0.07594188 0.07749472 0.07766465]

Residents LR classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2094
           2       0.16      0.01      0.02      5210
           3       0.17      0.27      0.21      6191
           4       0.16      0.76      0.26      6509
           5       0.00      0.00      0.00      4132
           6       0.00      0.00      0.00      3893
           7       0.33      0.00      0.00      4135
           8       0.14      0.00      0.01      4235
           9       0.00      0.00      0.00      3291
          10       0.00      0.00      0.00      1819
       micro avg   0.16      0.16      0.16     41509
       macro avg   0.10      0.10      0.05     41509
    weighted avg   0.12      0.16      0.08     41509

Residents LR confusion matrix :

    [[   0   24  592 1472    1    0    0    4    1    0]
     [   0   54 1519 3622    2    0    1    9    3    0]
     [   0   73 1672 4424    1    0    1   19    1    0]
     [   0   46 1494 4943    0    0    0   21    5    0]
     [   0   39 1047 3031    0    0    1   13    1    0]
     [   0   25  848 3009    0    0    0   10    1    0]
     [   0   19  849 3246    1    0    2   17    1    0]
     [   0   35  801 3379    0    0    1   19    0    0]
     [   0   18  552 2710    0    0    0   11    0    0]
     [   0    9  294 1504    0    0    0   12    0    0]]

##### SVM

Residents SVM 10CV f1_weighted scores : 
    
    [0.07583761 0.07603873 0.07671675 0.07697232 0.07674308 0.06844146 0.08025266 0.07703145 0.07713395 0.08787888]

Residents SVM classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2114
           2       0.18      0.01      0.01      5200
           3       0.17      0.31      0.22      6215
           4       0.16      0.72      0.26      6573
           5       0.00      0.00      0.00      4118
           6       0.00      0.00      0.00      3946
           7       0.00      0.00      0.00      4119
           8       0.14      0.00      0.00      4213
           9       0.00      0.00      0.00      3250
          10       0.00      0.00      0.00      1761
       micro avg   0.16      0.16      0.16     41509
       macro avg   0.06      0.10      0.05     41509
    weighted avg   0.09      0.16      0.08     41509

Residents SVM confusion matrix :

    [[   0    6  656 1448    0    0    0    2    2    0]
     [   0   38 1709 3445    0    0    0    6    2    0]
     [   0   42 1921 4240    0    0    0    5    7    0]
     [   0   35 1765 4763    0    0    1    7    2    0]
     [   0   22 1153 2940    0    0    0    2    1    0]
     [   0   23  996 2918    0    0    0    5    4    0]
     [   0   23  964 3130    0    0    0    2    0    0]
     [   0   12  930 3258    0    0    1    7    5    0]
     [   0    9  631 2599    0    0    0   11    0    0]
     [   0    6  363 1388    0    0    0    4    0    0]]

##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.08272591 0.08409931 0.08376657 0.08453753 0.08328807 0.08095945 0.0827335  0.08362372 0.08247099 0.08589722]

Residents XGBOOST classification report :

              precision    recall  f1-score   support
           1       0.00      0.00      0.00      2055
           2       0.19      0.03      0.06      5368
           3       0.17      0.33      0.23      6167
           4       0.16      0.70      0.25      6385
           5       0.12      0.00      0.00      4120
           6       0.00      0.00      0.00      4055
           7       0.19      0.00      0.00      4033
           8       0.25      0.00      0.00      4193
           9       0.00      0.00      0.00      3400
          10       0.00      0.00      0.00      1733
       micro avg   0.16      0.16      0.16     41509
       macro avg   0.11      0.11      0.05     41509
    weighted avg   0.13      0.16      0.08     41509

Residents XGBOOST confusion matrix :

    [[   0   69  755 1229    1    0    1    0    0    0]
     [   0  187 1826 3344    5    0    4    2    0    0]
     [   0  158 2057 3950    0    0    2    0    0    0]
     [   0  133 1791 4456    2    0    1    2    0    0]
     [   1  107 1231 2779    2    0    0    0    0    0]
     [   0   99 1077 2875    1    0    2    1    0    0]
     [   0   73  971 2983    1    0    4    1    0    0]
     [   0   74  961 3150    4    0    2    2    0    0]
     [   0   52  716 2629    0    0    3    0    0    0]
     [   0   30  371 1330    0    0    2    0    0    0]]

Residents XGBOOST features importances :

    [0.05112198 0.02648357 0.02144334 0.01846569 0.03395637 0.06794583
     0.02257886 0.04057742 0.03253445 0.0171969  0.02021412 0.02013488
     0.01873163 0.01920694 0.01865313 0.02139223 0.01853337 0.02048701
     0.01711538 0.02873799 0.0230107  0.02733737 0.02538207 0.01534694
     0.08300463 0.02576064 0.0256631  0.02313689 0.02113509 0.02072692
     0.02313739 0.02070953 0.0473064  0.03490462 0.01976109 0.02816556]

weight : 
    
    285 , 184 , 143 , 120 , 270 , 440 , 214 , 156 , 339 , 92 , 95 , 82 , 62 , 60 , 167 , 98 , 30 , 79 , 40 , 109 , 163 , 176 , 144 , 38 , 700 , 147 , 196 , 201 , 201 , 173 , 268 , 219 , 287 , 503 , 164 , 200 ,

gain : 

    17.17670620585964 , 8.898334996467398 , 7.204846734055945 , 6.204370729416668 , 11.409155395999983 , 22.8294277959523 , 7.586373605887855 , 13.633790981858974 , 10.93139664811209 , 5.778065959891304 , 6.7918354710526305 , 6.765211270268297 , 6.293726094838708 , 6.453426289116665 , 6.267350611856289 , 7.187672649999999 , 6.227111482000001 , 6.883524147848102 , 5.7506741709749996 , 9.65580718770642 , 7.731468328282207 , 9.185207675698866 , 8.528235482430558 , 5.15648803186842 , 27.889102384964264 , 8.655433275442176 , 8.622662241301017 , 7.773869004726371 , 7.101274408905474 , 6.964133463502896 , 7.77403549580224 , 6.958289981721463 , 15.894691175714293 , 11.727760129165008 , 6.639618365585366 , 9.463475085199995 ,

cover : 

    30407.005745414463 , 21661.735534061463 , 29249.00399142657 , 20468.737955433913 , 35981.56406505076 , 37596.647117162705 , 20972.331644119346 , 19995.537918214726 , 33575.13396717773 , 27085.579142067283 , 16893.942657024952 , 22581.046644529382 , 12398.630532440164 , 13128.24246746016 , 35996.45449838611 , 18141.68820362612 , 10825.438672032 , 18064.849755880885 , 24410.62025224175 , 36817.63476418239 , 28660.164529203306 , 28541.402619071992 , 23872.901814907716 , 34221.24581021105 , 35034.25056534935 , 26942.155154863267 , 24758.611411895152 , 23727.16532510614 , 32436.103338816712 , 33964.37907329826 , 33938.146217955145 , 28695.034561416818 , 27537.41058208416 , 30306.595607872638 , 23816.709922129394 , 27267.483730565706 ,

total_gain : 

    4895.361268669998 , 1637.293639350001 , 1030.2930829700001 , 744.5244875300002 , 3080.4719569199956 , 10044.948230219012 , 1623.483951660001 , 2126.87139317 , 3705.7434637099987 , 531.58206831 , 645.2243697499999 , 554.7473241620004 , 390.2110178799999 , 387.2055773469999 , 1046.6475521800003 , 704.3919196999999 , 186.81334446000002 , 543.7984076800001 , 230.02696683899998 , 1052.4829834599998 , 1260.2293375099998 , 1616.5965509230004 , 1228.0659094700004 , 195.94654521099997 , 19522.371669474986 , 1272.34869149 , 1690.0417992949992 , 1562.5476699500005 , 1427.3561561900003 , 1204.795089186001 , 2083.441512875 , 1523.8655059970004 , 4561.776367430002 , 5899.063344969999 , 1088.897411956 , 1892.6950170399991 ,

total_cover : 

    8665996.637443122 , 3985759.338267309 , 4182607.570773999 , 2456248.5546520697 , 9715022.297563706 , 16542524.73155159 , 4488078.97184154 , 3119303.9152414976 , 11381970.41487325 , 2491873.28107019 , 1604924.5524173705 , 1851645.8248514093 , 768715.0930112902 , 787694.5480476096 , 6011407.9012304805 , 1777885.4439553597 , 324763.16016096 , 1427123.13071459 , 976424.81008967 , 4013122.18929588 , 4671606.818260139 , 5023286.860956671 , 3437697.861346711 , 1300407.3407880198 , 24523975.395744544 , 3960496.8077649004 , 4852687.83673145 , 4769160.230346334 , 6519656.771102159 , 5875837.579680598 , 9095423.186411979 , 6284212.568950283 , 7903236.837058154 , 15244217.590759937 , 3905940.4272292205 , 5453496.746113141 ,
