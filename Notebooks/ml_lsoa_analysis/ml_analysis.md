
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

# XGBOOST importance types and feature names
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
features_names = ['f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17'\
                  ,'f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33'\
                  ,'f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49'\
                  ,'f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','f65'\
                  ,'f66','f67','f68','f69','f70','f71','f72','f73','f74','f75','f76','f77','f78','f79','f80','f81'\
                  ,'f82','f83','f84']

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

#### Logistic Regression
##### Antenna
Antenna LR 10CV f1_weighted scores : 

    [0.87560556 0.86190394 0.8790403  0.87427327 0.88034924 0.8714265 0.87127021 0.87174524 0.87368437 0.88281454]

Antenna LR classification report :

              precision    recall  f1-score   support
           1       0.95      0.87      0.91       184
           2       0.92      0.88      0.90       424
           3       0.74      0.93      0.83       601
           4       0.90      0.87      0.88       563
           5       0.90      0.85      0.87       356
           6       0.94      0.84      0.89       322
           7       0.91      0.91      0.91       358
           8       0.83      0.83      0.83       309
           9       0.92      0.77      0.84       198
          10       0.97      0.85      0.90       110
       micro avg   0.87      0.87      0.87      3425
       macro avg   0.90      0.86      0.88      3425
    weighted avg   0.88      0.87      0.87      3425

Antenna LR confusion matrix :

    [[160   7   9   6   1   0   0   1   0   0]
     [  3 375  28   1   3   2   3   7   2   0]
     [  2   4 561  11   1   6   2  12   2   0]
     [  2   2  46 490   6   2   5  10   0   0]
     [  1   4  21  17 301   4   3   3   2   0]
     [  0   7  22   7   5 271   6   1   3   0]
     [  0   4  18   4   4   1 324   3   0   0]
     [  0   1  22   8  11   2   5 256   3   1]
     [  0   2  23   3   1   1   6   8 152   2]
     [  0   0   6   0   1   0   3   6   1  93]]

##### Residents     
Residents LR 10CV f1_weighted scores : 

    [0.96798743 0.96579157 0.96786503 0.96762216 0.96658296 0.96084779 0.96096328 0.96272285 0.96168202 0.96286898]

Residents LR classification report :

              precision    recall  f1-score   support
           1       0.97      0.96      0.97      2042
           2       0.97      0.97      0.97      5259
           3       0.97      0.97      0.97      6259
           4       0.96      0.97      0.97      6468
           5       0.96      0.96      0.96      4141
           6       0.96      0.96      0.96      3998
           7       0.96      0.95      0.95      4047
           8       0.96      0.96      0.96      4264
           9       0.97      0.97      0.97      3267
          10       0.96      0.98      0.97      1764
       micro avg   0.96      0.96      0.96     41509
       macro avg   0.96      0.97      0.96     41509
    weighted avg   0.96      0.96      0.96     41509

Residents LR confusion matrix :

    [[1969   12   24   16    2    6    3    4    3    3]
     [  19 5103   51   27   22   14   15    4    4    0]
     [   8   37 6088   45   30   16   19   11    5    0]
     [  13   33   39 6263   29   30   23   21   12    5]
     [   2   11   29   39 3983   25   24   15   10    3]
     [   1   16   14   20   22 3835   30   34   17    9]
     [   6   26   15   38   22   19 3850   33   27   11]
     [   3   12   10   37   20   26   41 4073   24   18]
     [   5    0    7   19    5   17   14   21 3164   15]
     [   2    1    0    2    2    6    5   11    8 1727]]

#### SVM
##### Antenna
Antenna SVM 10CV f1_weighted scores : 

    [0.89269217 0.88251859 0.89405166 0.89423093 0.90364834 0.88929345 0.89833112 0.89546846 0.88895804 0.9092203 ]
 
Antenna SVM classification report :

              precision    recall  f1-score   support
           1       0.98      0.90      0.94       184
           2       0.90      0.93      0.92       424
           3       0.80      0.94      0.86       601
           4       0.92      0.89      0.90       563
           5       0.92      0.85      0.89       356
           6       0.97      0.86      0.91       322
           7       0.93      0.93      0.93       358
           8       0.86      0.89      0.87       309
           9       0.95      0.81      0.88       198
          10       0.95      0.91      0.93       110
       micro avg   0.90      0.90      0.90      3425
       macro avg   0.92      0.89      0.90      3425
    weighted avg   0.90      0.90      0.90      3425

Antenna SVM confusion matrix :

    [[165   4   7   6   1   0   0   1   0   0]
     [  1 393  15   4   0   1   0   9   0   1]
     [  1   8 565   8   1   1   2  11   4   0]
     [  0   7  39 500   5   3   3   6   0   0]
     [  0   7  24  11 304   2   3   3   2   0]
     [  0   9  11   9   5 277   6   3   1   1]
     [  0   3  18   1   2   0 333   1   0   0]
     [  0   2  16   3   9   0   3 274   1   1]
     [  1   2  11   4   3   1   7   6 161   2]
     [  0   0   4   0   0   0   1   5   0 100]]

##### Residents       
Residents SVM 10CV f1_weighted scores : 

    [0.96776669 0.96566857 0.96759747 0.96730572 0.9661951  0.96108891 0.96100706 0.96329884 0.96167917 0.96295981]

Residents SVM classification report :

              precision    recall  f1-score   support
           1       0.97      0.97      0.97      2042
           2       0.97      0.97      0.97      5259
           3       0.97      0.97      0.97      6259
           4       0.96      0.97      0.97      6468
           5       0.96      0.96      0.96      4141
           6       0.96      0.96      0.96      3998
           7       0.96      0.95      0.95      4047
           8       0.96      0.96      0.96      4264
           9       0.96      0.97      0.97      3267
          10       0.96      0.98      0.97      1764
       micro avg   0.96      0.96      0.96     41509
       macro avg   0.96      0.97      0.96     41509
    weighted avg   0.96      0.96      0.96     41509

Residents SVM confusion matrix :

    [[1971   12   23   16    2    6    3    4    3    2]
     [  18 5106   48   27   20   15   17    4    4    0]
     [  10   36 6083   45   28   17   17   16    6    1]
     [  13   31   40 6261   28   31   22   22   14    6]
     [   3   13   30   40 3976   26   23   16   10    4]
     [   1   19   16   19   22 3827   28   36   19   11]
     [   6   25   15   42   24   17 3840   35   30   13]
     [   3   11   10   33   20   22   36 4085   25   19]
     [   3    0    7   16    5   18   15   17 3173   13]
     [   2    0    0    3    1    3    5   11    7 1732]]

#### XGBOOST
##### Antenna
Antenna XGBOOST 10CV f1_weighted scores : 

    [0.9123163  0.8990322  0.90395565 0.90791748 0.91131416 0.91159137 0.91701889 0.90934044 0.90627642 0.92199817]
    
Antenna XGBOOST classification report :

              precision    recall  f1-score   support
           1       0.97      0.92      0.94       184
           2       0.91      0.94      0.92       424
           3       0.86      0.95      0.90       601
           4       0.92      0.89      0.91       563
           5       0.93      0.89      0.91       356
           6       0.92      0.90      0.91       322
           7       0.92      0.92      0.92       358
           8       0.92      0.91      0.91       309
           9       0.95      0.88      0.92       198
          10       0.96      0.90      0.93       110
       micro avg   0.91      0.91      0.91      3425
       macro avg   0.93      0.91      0.92      3425
    weighted avg   0.92      0.91      0.91      3425

Antenna XGBOOST confusion matrix :

    [[169   2   7   3   1   2   0   0   0   0]
     [  2 398  16   3   0   1   4   0   0   0]
     [  2   4 568  14   4   4   4   1   0   0]
     [  1   9  25 502   6   7   8   5   0   0]
     [  0   7  15   7 317   4   2   3   1   0]
     [  0   5  10   3   4 291   4   2   3   0]
     [  0   3   9   4   4   2 331   3   1   1]
     [  0   8   6   4   6   0   1 281   3   0]
     [  0   3   2   3   0   3   4   5 175   3]
     [  0   0   0   0   0   1   2   7   1  99]]
 
Antenna XGBOOST features importances :

    [0.03268632 0.14242674 0.00467983 0.00337669 0.0040314  0.05773998 0.00734292 0.00292725 0.00577084 0.00492294 0.00315223 0.00240144 0.0208094  0.05437677 0.00686406 0.07565753 0.00767443 0.06449085 0.00941238 0.06599484 0.0151742  0.0691306  0.00937424 0.05926599 0.01262944 0.06657354 0.00917883 0.06747419 0.0086686  0.04753536 0.00670662 0.04712773 0.00442184]
 
weight : 

    80 , 197 , 37 , 19 , 23 , 114 , 20 , 27 , 67 , 81 , 52 , 37 , 91 , 363 , 100 , 548 , 128 , 671 , 182 , 612 , 118 , 470 , 147 , 460 , 99 , 473 , 168 , 467 , 104 , 469 , 150 , 346 , 67 ,
    
gain : 

    56.191974559499954 , 244.84983151056358 , 8.045232576486487 , 5.80496657631579 , 6.930485333720868 , 99.26244479296489 , 12.6234146356 , 5.0323170416296295 , 9.920808493865673 , 8.463166855148149 , 5.419089456538463 , 4.128388249864865 , 35.77402804623188 , 93.48063839510733 , 11.800198810290997 , 130.06499729421884 , 13.193323318547659 , 110.86803426797249 , 16.181095290714293 , 113.45360591244442 , 26.086389972822023 , 118.84437205157445 , 16.115527829564634 , 101.88584879175231 , 21.71162660534343 , 114.44845090204429 , 15.779595567976184 , 115.99678015674733 , 14.902442771932696 , 81.71939036695096 , 11.529544928360004 , 81.01861320678034 , 7.601704948194033 ,
    
cover :

    410.4662039862499 , 1120.8948241543155 , 306.9599592027027 , 292.9625881978947 , 322.5001178826088 , 664.4240086561405 , 581.69286094 , 226.98347665185182 , 397.5307958928358 , 404.85506204938287 , 148.8672792788462 , 202.10170799540535 , 428.5557035318682 , 987.3583449609653 , 381.7559160329999 , 1333.2737948712231 , 823.4165448515621 , 1485.258258862742 , 1194.327029276374 , 1490.9355618311606 , 781.6618871331355 , 1362.331128081914 , 558.0884327129253 , 1298.121439979434 , 625.3500303090909 , 1296.0724075652008 , 950.5788100683931 , 1240.5197413725912 , 643.9888035173078 , 957.7855335970139 , 320.92160033580006 , 803.8555597635838 , 295.04071672388056 ,
    
total_gain : 

    4495.357964759996 , 48235.41680758102 , 297.67360533000004 , 110.29436495000002 , 159.40116267557997 , 11315.918706397997 , 252.468292712 , 135.872560124 , 664.6941690890001 , 685.516515267 , 281.79265174000005 , 152.750365245 , 3255.4365522071007 , 33933.47173742396 , 1180.0198810290997 , 71275.61851723192 , 1688.7453847741003 , 74392.45099380954 , 2944.959342910001 , 69433.60681841598 , 3078.194016792999 , 55856.85486423999 , 2368.982590946001 , 46867.490444206065 , 2149.4510339289995 , 54134.11727666695 , 2650.972055419999 , 54170.49633320101 , 1549.8540482810004 , 38326.3940821 , 1729.4317392540006 , 28032.440169546 , 509.3142315290002 ,
    
total_cover : 

    32837.29631889999 , 220816.28035840017 , 11357.518490499999 , 5566.28917576 , 7417.5027113000015 , 75744.33698680002 , 11633.8572188 , 6128.553869599999 , 26634.563324819996 , 32793.26002600001 , 7741.098522500003 , 7477.763195829998 , 38998.569021400006 , 358411.0792208304 , 38175.59160329999 , 730634.0395894303 , 105397.31774099995 , 996608.2916968998 , 217367.51932830008 , 912452.5638406703 , 92236.10268170998 , 640295.6301984995 , 82038.99960880002 , 597135.8623905396 , 61909.6530006 , 613042.2487783399 , 159697.24009149004 , 579322.7192210001 , 66974.83556580001 , 449201.41525699955 , 48138.24005037001 , 278134.02367819997 , 19767.7280205 ,

##### Residents       
Residents XGBOOST 10CV f1_weighted scores : 

    [0.96872684 0.96667148 0.96892066 0.96934734 0.96814007 0.96250368 0.96228457 0.96391869 0.96409007 0.96442005]
    
Residents XGBOOST classification report :

              precision    recall  f1-score   support
           1       0.97      0.97      0.97      2042
           2       0.97      0.97      0.97      5259
           3       0.97      0.97      0.97      6259
           4       0.96      0.97      0.97      6468
           5       0.97      0.96      0.97      4141
           6       0.96      0.96      0.96      3998
           7       0.96      0.95      0.96      4047
           8       0.97      0.96      0.96      4264
           9       0.97      0.97      0.97      3267
          10       0.96      0.98      0.97      1764
       micro avg   0.97      0.97      0.97     41509
       macro avg   0.97      0.97      0.97     41509
    weighted avg   0.97      0.97      0.97     41509

Residents XGBOOST confusion matrix :

    [[1983   10   20   14    2    3    2    4    3    1]
     [  17 5110   46   27   22   14   16    4    3    0]
     [  10   42 6098   41   20   16   13   15    4    0]
     [  13   35   42 6290   16   31   18   12    5    6]
     [   3   11   30   42 3981   27   23   12    9    3]
     [   1   18   16   24   14 3856   21   28   15    5]
     [   6   28   15   39   22   17 3835   39   29   17]
     [   3   10   10   41   17   26   29 4078   30   20]
     [   3    0    9   15    5   18   16   21 3165   15]
     [   2    1    1    3    0    5    4   11    8 1729]]
     
Residents XGBOOST features importances :

    [6.82199970e-05 6.89071967e-05 2.52253609e-04 1.96218185e-04 6.42929881e-05 1.65089572e-04 5.85557464e-05 2.50781159e-04 1.68325103e-04 2.19171590e-04 8.15481471e-05 3.18159146e-04 4.32823290e-05 5.69525873e-05 0.00000000e+00 6.59677607e-05 1.86850317e-04 6.09745621e-05 2.64520495e-04 2.33945451e-04 1.27820866e-04 1.90262537e-04 2.55450374e-04 0.00000000e+00 3.00962158e-04 9.09784576e-05 2.26103512e-04 1.68897925e-04 1.06558720e-04 1.46883176e-04 1.73084598e-04 1.93747634e-04 1.44306614e-04 1.25068065e-04 1.66601560e-04 1.68752929e-04 5.67767071e-04 9.15151497e-04 4.70968895e-04 6.00684492e-04 3.54930729e-04 4.52621316e-04 2.09484715e-04 3.60010192e-04 2.67946685e-04 1.23878941e-04 1.05716703e-04 2.10570230e-04 3.28408089e-04 1.46965293e-04 2.82478170e-04 2.21806782e-04 2.94619036e-04 1.14812050e-04 1.58940849e-04 2.53404985e-04 3.06809583e-04 1.40417294e-04 2.19935173e-04 3.14163422e-04 6.60143342e-05 6.84900297e-05 4.20736033e-04 1.99378323e-04 8.30682329e-05 5.03900264e-05 1.01650803e-04 2.67373340e-04 2.57404026e-04 4.04197526e-05 1.60814321e-04 6.49055364e-05 6.54119402e-02 1.15258545e-01 1.21726893e-01 1.32780850e-01 1.05203182e-01 9.25960466e-02 9.06388238e-02 9.76205692e-02 1.00888737e-01 5.39850891e-02 2.33994145e-03 6.93669682e-03]
    
weight : 

    5 , 50 , 3 , 17 , 23 , 10 , 56 , 2 , 19 , 3 , 4 , 10 , 5 , 1 , None , 19 , 5 , 14 , 9 , 7 , 5 , 29 , 7 , None , 12 , 7 , 25 , 15 , 21 , 17 , 39 , 8 , 3 , 30 , 13 , 15 , 30 , 29 , 32 , 9 , 20 , 29 , 44 , 9 , 12 , 3 , 20 , 7 , 3 , 5 , 6 , 6 , 3 , 14 , 2 , 6 , 1 , 5 , 3 , 1 , 94 , 37 , 7 , 5 , 29 , 24 , 28 , 12 , 45 , 60 , 11 , 15 , 356 , 386 , 401 , 375 , 354 , 391 , 391 , 392 , 324 , 376 , 723 , 840 ,
    
gain : 
    
    1.5195430752 , 1.5348497391859999 , 5.6187365866666665 , 4.37059498 , 1.4320721627826087 , 3.6772312629000004 , 1.3042799417267859 , 5.585938695 , 3.749299952473684 , 4.881862483333333 , 1.81641642675 , 7.0867272797 , 0.9640774482000001 , 1.2685709 , None , 1.4693762752105262 , 4.161933517 , 1.3581570897857145 , 5.891971332000001 , 5.2109377888571435 , 2.8471021668 , 4.237938147275862 , 5.689942017142857 , None , 6.7036784475 , 2.0264684692857142 , 5.036265279279999 , 3.762058985840667 , 2.3735058970000003 , 3.271699212647059 , 3.8553138630769226 , 4.315565735 , 3.214308223333333 , 2.7857859927329995 , 3.710909697230769 , 3.7588293478666666 , 12.646533846666669 , 20.384229062586204 , 10.490435642187501 , 13.379741877777777 , 7.905783031999998 , 10.081758838965516 , 4.666095922181818 , 8.018924023333334 , 5.96828697325 , 2.7592990386666667 , 2.3547506919499996 , 4.690274477142857 , 7.315013726666667 , 3.2735280520000005 , 6.291963576666666 , 4.9405595485000005 , 6.562390646666667 , 2.557341779857143 , 3.540273668 , 5.644382633833334 , 6.83392525 , 3.127676964 , 4.898871103333334 , 6.99772549 , 1.4704136694031915 , 1.525557711135135 , 9.371540656285715 , 4.44098425 , 1.850275015344828 , 1.1223954483333334 , 2.264186077392857 , 5.955515660833334 , 5.733457888444444 , 0.9003160274633333 , 3.5820036063636365 , 1.4457161469333333 , 1456.9959614750974 , 2567.2870066420005 , 2711.3640583175074 , 2957.5817265652445 , 2343.3121931209594 , 2062.4990145265283 , 2018.9034550705599 , 2174.416063031278 , 2247.211777432025 , 1202.472385196336 , 52.12022710557278 , 154.50907376573704 ,
    
cover : 

    2523.2794446400003 , 882.034871704 , 2114.237383333333 , 786.5554823111764 , 1388.0498853043475 , 608.1167953649999 , 1314.503468103572 , 1746.9571765 , 2256.4479824210525 , 267.89954623333335 , 74.90817577 , 3608.3151882000006 , 92.29904186000002 , 5.81977224 , None , 334.65780133157887 , 2367.642938 , 521.2540633142856 , 3564.3814262077776 , 267.88290507285717 , 735.9047997719999 , 2213.4091981982756 , 1375.7738543285716 , None , 1959.473649325 , 663.1579140714286 , 806.7402886640001 , 664.5468139866667 , 190.13526287999997 , 1051.3980368735297 , 406.37451771538457 , 807.0190037 , 125.92596776 , 961.7629860893334 , 794.584352469231 , 1102.0617982666668 , 1401.7096522233333 , 1052.1108945348276 , 790.7666572437499 , 627.842198211111 , 3939.0283496799993 , 331.2325074068965 , 1750.9684715027274 , 272.3800568555555 , 2160.16580653 , 55.295926060000006 , 617.61493595 , 1177.1898211571427 , 1375.236163333333 , 1110.72237638 , 688.3305258666668 , 4233.637696666668 ,5382.246910000001 , 1757.5087703857141 , 1894.9068915 , 2933.6076861666666 , 2604.10205 , 31.06636352 , 1110.34241585 , 490.658752 , 1857.2855084587231 , 1703.7940421513515 , 665.2417680971429 , 1900.559536824 , 155.5069047910345 , 866.9414541954167 , 686.810180375 , 1639.4041883583332 , 1145.0544906355553 , 862.21240336 , 878.5991330527272 , 123.16046663399999 , 10637.27551225561 , 12625.812611273144 , 13136.42542539719 , 14274.335663278423 , 12345.023242479225 , 11313.050228601538 , 11458.756430186171 , 11134.825201325517 , 12404.708401820983 , 9694.67137580197 , 4752.860872962246 , 1174.9086417085473 ,
    
total_gain : 

    7.597715376 , 76.7424869593 , 16.85620976 , 74.30011466 , 32.937659744 , 36.772312629000005 , 73.0396767367 , 11.17187739 , 71.236699097 , 14.64558745 , 7.265665707 , 70.867272797 , 4.820387241000001 , 1.2685709 , None , 27.918149228999997 , 20.809667585 , 19.014199257 , 53.027741988 , 36.476564522000004 , 14.235510834 , 122.900206271 , 39.82959412 , None , 80.44414137 , 14.185279285 , 125.90663198199998 , 56.430884787610005 , 49.84362383700001 , 55.618886615 , 150.35724065999997 , 34.52452588 , 9.64292467 , 83.57357978198999 , 48.241826064 , 56.382440218 , 379.39601540000007 , 591.1426428149999 , 335.69394055000004 , 120.41767689999999 , 158.11566063999996 , 292.37100633 , 205.30822057600002 , 72.17031621 , 71.619443679 , 8.277897116 , 47.095013838999996 , 32.83192134 , 21.94504118 , 16.36764026 , 37.75178146 , 29.643357291 , 19.68717194 , 35.802784918 , 7.080547336 , 33.866295803 , 6.83392525 , 15.63838482 , 14.69661331 , 6.99772549 , 138.2188849239 , 56.445635312 , 65.600784594 , 22.204921249999998 , 53.65797544500001 , 26.937490760000003 , 63.397210167 , 71.46618793 , 258.00560498 , 54.0189616478 , 39.40203967 , 21.685742204 , 518690.5622851346 , 990972.7845638122 , 1087256.9873853205 , 1109093.1474619666 , 829532.5163648196 , 806437.1146798725 , 789391.2509325889 , 852371.0967082611 , 728096.6158879761 , 452129.6168338224 , 37682.92419732912 , 129787.62196321912 ,
    
total_cover : 

    12616.397223200001 , 44101.7435852 , 6342.712149999999 , 13371.443199289999 , 31925.147361999992 , 6081.16795365 , 73612.19421380003 , 3493.914353 , 42872.511666 , 803.6986387000001 , 299.63270308 , 36083.151882000006 , 461.49520930000006 , 5.81977224 , None , 6358.498225299998 , 11838.21469 , 7297.5568864 , 32079.43283587 , 1875.18033551 , 3679.5239988599997 , 64188.86674775 , 9630.4169803 , None , 23513.6837919 , 4642.105398500001 , 20168.507216600003 , 9968.202209800002 , 3992.8405204799997 , 17873.766626850003 , 15848.606190899998 , 6456.1520296 , 377.77790328000003 , 28852.88958268 , 10329.596582100003 , 16530.926974 , 42051.2895667 , 30511.215941510003 , 25304.533031799998 , 5650.5797839 , 78780.56699359999 , 9605.742714799999 , 77042.61274612001 , 2451.4205116999997 , 25921.989678359998 , 165.88777818000003 , 12352.298719 , 8240.328748099999 , 4125.708489999999 , 5553.6118819 , 4129.9831552000005 , 25401.826180000004 , 16146.740730000001 , 24605.1227854 , 3789.813783 , 17601.646117 , 2604.10205 , 155.3318176 , 3331.02724755 , 490.658752 , 174584.83779511997 , 63040.379559600005 , 4656.69237668 , 9502.79768412 , 4509.70023894 , 20806.59490069 , 19230.6850505 , 19672.850260299998 , 51527.45207859999 , 51732.744201600006 , 9664.59046358 , 1847.4069995099999 , 3786870.0823629973 , 4873563.667951434 , 5267706.595584273 , 5352875.873729409 , 4370138.227837645 , 4423402.6393832015 , 4480373.764202793 , 4364851.478919603 , 4019125.522189998 , 3645196.437301541 , 3436318.4111517034 , 986923.2590351797 ,
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
    
## 4.3 Binary LSOA IMD Feature subsets - RICH(8,9,10) vs POOR(1,2,3)
```python
residents_dt_pd_ml_scaled['LSOA_IMD_decile'][residents_dt_pd_ml_scaled['LSOA_IMD_decile']<=3] = 0
residents_dt_pd_ml_scaled['LSOA_IMD_decile'][residents_dt_pd_ml_scaled['LSOA_IMD_decile']>=8] = 1
residents_dt_pd_ml_scaled = residents_dt_pd_ml_scaled[residents_dt_pd_ml_scaled['LSOA_IMD_decile'].isin([0,1])]

antenna_dt_pd_ml_scaled['LSOA_IMD_decile'][antenna_dt_pd_ml_scaled['LSOA_IMD_decile']<=3] = 0
antenna_dt_pd_ml_scaled['LSOA_IMD_decile'][antenna_dt_pd_ml_scaled['LSOA_IMD_decile']>=8] = 1
antenna_dt_pd_ml_scaled = antenna_dt_pd_ml_scaled[antenna_dt_pd_ml_scaled['LSOA_IMD_decile'].isin([0,1])]
```

### 4.3.1 Antenna feature subsets
#### 4.3.1.1 antenna_feature_subset_sector
##### Logistic Regression

Antenna LR 10CV f1_weighted scores : 

  [0.70886982 0.73294302 0.73450277 0.73412011 0.74764096 0.70604187 0.73031323 0.73814431 0.72639077 0.71864827]
  
Antenna LR classification report :

              precision    recall  f1-score   support
           0       0.78      0.89      0.83      1250
           1       0.67      0.47      0.55       601
       micro avg   0.75      0.75      0.75      1851
       macro avg   0.72      0.68      0.69      1851
    weighted avg   0.74      0.75      0.74      1851

Antenna LR confusion matrix :

    [[1110  140]
     [ 320  281]]

##### SVM

Antenna SVM 10CV f1_weighted scores : 

  [0.70052878 0.72674801 0.73529802 0.73010577 0.74183965 0.70434084 0.7274311  0.73406711 0.72181482 0.71761447]
  
Antenna SVM classification report :

              precision    recall  f1-score   support
           0       0.77      0.89      0.83      1250
           1       0.67      0.45      0.54       601
       micro avg   0.75      0.75      0.75      1851
       macro avg   0.72      0.67      0.68      1851
    weighted avg   0.74      0.75      0.73      1851

Antenna SVM confusion matrix :

    [[1118  132]
     [ 331  270]]

##### XGBOOST

Antenna XGBOOST 10CV f1_weighted scores : 

    [0.72493712 0.73969666 0.76001104 0.7533386  0.75150826 0.73278991 0.73903814 0.76112798 0.75418796 0.75498983]
  
Antenna XGBOOST classification report :

              precision    recall  f1-score   support
           0       0.79      0.88      0.84      1250
           1       0.68      0.52      0.59       601
       micro avg   0.76      0.76      0.76      1851
       macro avg   0.74      0.70      0.71      1851
    weighted avg   0.76      0.76      0.75      1851

Antenna XGBOOST confusion matrix :

    [[1104  146]
     [ 290  311]]
 
Antenna XGBOOST features importances :

    [0.05656455 0.07578638 0.1996561  0.16373539 0.05112124 0.04263685  0.03372309 0.06433333 0.04660424 0.03909919 0.04869572 0.12761253 0.05043146]
 
weight : 

    36 , 18 , 112 , 68 , 43 , 15 , 51 , 93 , 42 , 19 , 36 , 80 , 79 ,
    
gain : 

    25.048884299166662 , 33.56102861055555 , 88.41514106339282 , 72.50811551926473 , 22.63838376767442 , 18.881181082666668 , 14.933835189019609 , 28.489188559634407 , 20.638090605952378 , 17.314574391578947 , 21.5642744435 , 56.511572994125025 , 22.332923757848103 ,
    
cover : 

    1466.8982235916671 , 1007.6819738505557 , 1909.64688714375 , 1452.072639628382 , 1465.598155983721 , 492.5926409333333 , 890.6728899960787 , 1206.8491023749464 , 1513.8011346309522 , 1471.3579706157896 , 835.9301706822222 , 1528.2844707555 , 1376.050199391519 ,
  
total_gain : 

    901.7598347699999 , 604.09851499 , 9902.495799099996 , 4930.551855310002 , 973.45050201 , 283.21771624 , 761.62559464 , 2649.4945360459997 , 866.7998054499999 , 328.97691344000003 , 776.3138799660001 , 4520.925839530002 , 1764.30097687 ,
    
total_cover : 

    52808.336049300015 , 18138.27552931 , 213880.4513601 , 98740.93949472997 , 63020.7207073 , 7388.889614 , 45424.31738980001 , 112236.96652087002 , 63579.64765449999 , 27955.8014417 , 30093.48614456 , 122262.75766044 , 108707.96575193001 ,

### 4.3.2 Residents feature subsets
#### 4.3.2.1 residents_feature_subset_perf34g
##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.5100847  0.51169743 0.50847228 0.51083838 0.50845998 0.51225027 0.51340859 0.51889486 0.5144396  0.51536935]
 
Residents LR classification report :

              precision    recall  f1-score   support
           0       0.61      0.93      0.74     13632
           1       0.53      0.12      0.20      9214
       micro avg   0.60      0.60      0.60     22846
       macro avg   0.57      0.52      0.47     22846
    weighted avg   0.58      0.60      0.52     22846

Residents LR confusion matrix :

    [[12667   965]
     [ 8106  1108]]
 
Residents SVM 10CV f1_weighted scores : 

    [0.51552186 0.50964279 0.507217   0.51322271 0.51043316 0.51585084 0.51638107 0.5186569  0.51750591 0.52083355]

##### SVM

Residents SVM classification report :

              precision    recall  f1-score   support
           0       0.61      0.93      0.74     13632
           1       0.53      0.12      0.20      9214
       micro avg   0.60      0.60      0.60     22846
       macro avg   0.57      0.52      0.47     22846
    weighted avg   0.58      0.60      0.52     22846

Residents SVM confusion matrix :

    [[12653   979]
     [ 8096  1118]]

##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.5476617  0.54873955 0.54549849 0.54516714 0.55392688 0.55416314 0.55238797 0.55583693 0.554325   0.55387022]
 
Residents XGBOOST classification report :

              precision    recall  f1-score   support
           0       0.62      0.89      0.73     13632
           1       0.55      0.20      0.30      9214
       micro avg   0.61      0.61      0.61     22846
       macro avg   0.59      0.55      0.51     22846
    weighted avg   0.59      0.61      0.56     22846

Residents XGBOOST confusion matrix :

    [[12087  1545]
     [ 7335  1879]]
 
Residents XGBOOST features importances :

    [0.00941689 0.00950925 0.00412035 0.01553276 0.01409162 0.00691679
     0.00982357 0.00990484 0.01250425 0.01014016 0.         0.00621607
     0.00377565 0.00915621 0.01965201 0.00370649 0.00441238 0.00489952
     0.         0.         0.0180655  0.02035736 0.01213279 0.01116758
     0.01821331 0.00640329 0.03919489 0.00740397 0.00852848 0.00580618
     0.0106399  0.01406694 0.00966152 0.00932486 0.00493315 0.0058277
     0.03629246 0.01136482 0.00935005 0.00524354 0.0192734  0.05698798
     0.00907678 0.09524863 0.02982962 0.00505581 0.00505736 0.00627936
     0.00523788 0.         0.00653998 0.00863629 0.00679196 0.00499577
     0.00650168 0.01131796 0.01119984 0.01616206 0.0128454  0.00769648
     0.08889049 0.01157073 0.00646355 0.00834962 0.01538039 0.00995307
     0.01463944 0.00868847 0.04021126 0.0333829  0.00887521 0.0111035 ]

weight : 

    6 , 4 , 2 , 7 , 19 , 4 , 17 , 6 , 15 , 1 , None , 1 , 2 , 2 , 16 , 1 , 1 , 3 , None , None , 9 , 22 , 8 , 2 , 16 , 6 , 4 , 1 , 4 , 4 , 12 , 9 , 14 , 14 , 2 , 4 , 23 , 19 , 2 , 2 , 21 , 43 , 20 , 7 , 22 , 2 , 6 , 1 , 2 , None , 11 , 5 , 2 , 1 , 3 , 11 , 9 , 24 , 13 , 1 , 64 , 3 , 7 , 7 , 10 , 9 , 20 , 4 , 25 , 47 , 7 , 9 ,

gain : 

    26.834301783333334 , 27.09746695 , 11.741308199999999 , 44.26202224285714 , 40.15535785368421 , 19.7100296075 , 27.993156652352948 , 28.22475401833334 , 35.63200532 , 28.8953247 , None , 17.7132587 , 10.7590518 , 26.09144405 , 56.000189400000004 , 10.5619726 , 12.5734797 , 13.961629866666664 , None , None , 51.47928158888889 , 58.01015017272727 , 34.5734786875 , 31.823038099999998 , 51.90047691874999 , 18.246769733333334 , 111.689412875 , 21.0983028 , 24.302687625 , 16.545242899999998 , 30.319375900833332 , 40.08502006666667 , 27.53138747142857 , 26.57204531428571 , 14.0574622 , 16.6065526 , 103.41868209130435 , 32.385090878947366 , 26.643830299999998 , 14.9419527 , 54.921307571428564 , 162.3924532 , 25.865122125 , 271.41967092857146 , 85.00222581363633 , 14.406990050000001 , 14.411398908333334 , 17.8935986 , 14.9258051 , None , 18.636261854545452 , 24.609909827999996 , 19.354314849999998 , 14.2359123 , 18.52712186666667 , 32.25156332727273 , 31.914958309999992 , 46.05527433708334 , 36.60413666923077 , 21.9318314 , 253.30157607265622 , 32.97185006666667 , 18.41846276 , 23.793015057142856 , 43.827823632000005 , 28.362188444444445 , 41.716425795 , 24.7585993 , 114.585661828 , 95.1276132914894 , 25.29071249857143 , 31.640436488888895 ,

cover : 

    42200.93556666667 , 24788.0211075 , 374.4860077 , 16993.096465714283 , 26293.60776610526 , 13050.972250750001 , 14265.904746999997 , 13398.974965466667 , 22675.647227946665 , 3619.34033 , None , 1307.43958 , 92.1494769 , 32286.8252 , 30203.238120875 , 934.884277 , 560.34967 , 271.9652456666667 , None , None , 25285.442971666667 , 23413.686140590908 , 12192.62701625 , 25708.064449999998 , 20730.892106499996 , 17688.699321166667 , 21197.2403475 , 5614.06104 , 5710.1768375 , 8382.4171648025 , 9386.365380916666 , 31206.785372222224 , 23035.456355928567 , 17928.507426 , 284.4716224 , 2605.82487525 , 23962.898016086958 , 15138.99474494737 , 12290.4995 , 24040.432855 , 25280.285918085712 , 25152.3994565814 , 10481.64409425 , 24300.77866857143 , 23660.39090140909 , 1185.803862 , 3326.8908969999998 , 48146.1172 , 540.63208 , None , 27832.373546710005 , 23016.356351839997 , 480.6544795 , 1039.80737 , 46165.918 , 31865.505631300002 , 14955.64478322222 , 16974.435870587502 , 15471.092983076924 , 41934.1445 , 25134.601059953122 , 13039.13203 , 11853.185511714286 , 4790.642447714286 , 27725.838030100003 , 25453.221569777776 , 28945.7035886 , 25052.651723000003 , 16365.934719239995 , 25080.773772765955 , 9667.450479714287 , 15707.176131111113 ,

total_gain : 

    161.0058107 , 108.3898678 , 23.482616399999998 , 309.8341557 , 762.95179922 , 78.84011843 , 475.8836630900001 , 169.34852411000003 , 534.4800798 , 28.8953247 , None , 17.7132587 , 21.5181036 , 52.1828881 , 896.0030304000001 , 10.5619726 , 12.5734797 , 41.884889599999994 , None , None , 463.3135343 , 1276.2233038 , 276.5878295 , 63.646076199999996 , 830.4076306999998 , 109.48061840000001 , 446.7576515 , 21.0983028 , 97.2107505 , 66.18097159999999 , 363.83251081 , 360.76518060000006 , 385.4394246 , 372.00863439999995 , 28.1149244 , 66.4262104 , 2378.6296881 , 615.3167267 , 53.287660599999995 , 29.8839054 , 1153.3474589999998 , 6982.8754876 , 517.3024425 , 1899.9376965000004 , 1870.0489678999993 , 28.813980100000002 , 86.46839345000001 , 17.8935986 , 29.8516102 , None , 204.9988804 , 123.04954913999998 , 38.708629699999996 , 14.2359123 , 55.581365600000005 , 354.76719660000003 , 287.23462478999994 , 1105.32658409 , 475.85377669999997 , 21.9318314 , 16211.300868649998 , 98.9155502 , 128.92923932 , 166.55110539999998 , 438.2782363200001 , 255.25969600000002 , 834.3285159 , 99.0343972 , 2864.6415457 , 4470.997824700002 , 177.03498749000002 , 284.76392840000005 ,

total_cover : 

    253205.6134 , 99152.08443 , 748.9720154 , 118951.67525999999 , 499578.54755599995 , 52203.889003000004 , 242520.38069899994 , 80393.8497928 , 340134.7084192 , 3619.34033 , None , 1307.43958 , 184.2989538 , 64573.6504 , 483251.809934 , 934.884277 , 560.34967 , 815.895737 , None , None , 227568.986745 , 515101.095093 , 97541.01613 , 51416.128899999996 , 331694.27370399993 , 106132.195927 , 84788.96139 , 5614.06104 , 22840.70735 , 33529.66865921 , 112636.384571 , 280861.06835 , 322496.38898299995 , 250999.10396399998 , 568.9432448 , 10423.299501 , 551146.65437 , 287640.90015400003 , 24580.999 , 48080.86571 , 530886.0042798 , 1081553.1766330001 , 209632.88188499998 , 170105.45068 , 520528.59983099997 , 2371.607724 , 19961.345382 , 48146.1172 , 1081.26416 , None , 306156.10901381006 , 115081.78175919998 , 961.308959 , 1039.80737 , 138497.754 , 350520.5619443 , 134600.80304899998 , 407386.4608941 , 201124.20878000002 , 41934.1445 , 1608614.4678369998 , 39117.39609 , 82972.298582 , 33534.497134000005 , 277258.380301 , 229078.994128 , 578914.071772 , 100210.60689200001 , 409148.3679809999 , 1178796.36732 , 67672.15335800001 , 141364.58518000002 ,

#### 4.3.2.2 residents_feature_subset_perf3g
##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.45350556 0.45711529 0.45502734 0.45360884 0.45171625 0.44890725 0.44877391 0.44931103 0.44838562 0.44794562]

Residents LR classification report :

              precision    recall  f1-score   support
           0       0.59      0.99      0.74     13499
           1       0.44      0.01      0.03      9347
       micro avg   0.59      0.59      0.59     22846
       macro avg   0.51      0.50      0.38     22846
    weighted avg   0.53      0.59      0.45     22846

Residents LR confusion matrix :

    [[13324   175]
     [ 9212   135]]

##### SVM

Residents SVM 10CV f1_weighted scores : 

    [0.44584145 0.44590983 0.44524988 0.44670334 0.44426326 0.44443126 0.44330355 0.44515011 0.44493944 0.44372373]
 
Residents SVM classification report :

              precision    recall  f1-score   support
           0       0.59      1.00      0.74     13499
           1       0.54      0.01      0.01      9347
       micro avg   0.59      0.59      0.59     22846
       macro avg   0.57      0.50      0.38     22846
    weighted avg   0.57      0.59      0.44     22846

Residents SVM confusion matrix :

    [[13446    53]
     [ 9285    62]]

##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.46827278 0.47298003 0.46815782 0.46873839 0.46134285 0.45957066 0.4592559  0.45968911 0.46295214 0.46018481]

Residents XGBOOST classification report :

              precision    recall  f1-score   support
           0       0.59      0.97      0.74     13499
           1       0.50      0.04      0.07      9347
       micro avg   0.59      0.59      0.59     22846
       macro avg   0.55      0.51      0.40     22846
    weighted avg   0.56      0.59      0.46     22846

Residents XGBOOST confusion matrix :

    [[13156   343]
     [ 9002   345]]
 
Residents XGBOOST features importances :

    [0.05567497 0.05661258 0.01207603 0.05144129 0.02682432 0.04008013
     0.01248949 0.02355919 0.01173297 0.0070828  0.04125876 0.01787337
     0.01580969 0.01016596 0.0409693  0.01462991 0.00983383 0.00934485
     0.00654198 0.00999486 0.01616932 0.03115198 0.02656252 0.01099593
     0.08629861 0.03076494 0.03514945 0.03816332 0.15765794 0.0165373
     0.01287642 0.01235933 0.01007297 0.02104119 0.01333969 0.00686264]

weight : 

    22 , 24 , 12 , 20 , 49 , 28 , 26 , 14 , 23 , 10 , 11 , 16 , 3 , 4 , 23 , 6 , 4 , 12 , 6 , 4 , 25 , 24 , 26 , 7 , 63 , 23 , 41 , 6 , 8 , 24 , 31 , 15 , 18 , 45 , 14 , 10 ,

gain : 

    75.90927108886363 , 77.18763805625 , 16.464896452500003 , 70.13691645999998 , 36.57324315040815 , 54.64669710428571 , 17.028626964615388 , 32.12145347071428 , 15.997155636086955 , 9.656940602 , 56.253688635454544 , 24.369198526875 , 21.555506366666666 , 13.860640974999999 , 55.85902508652175 , 19.946945595000003 , 13.4078047375 , 12.741100147499997 , 8.919567825 , 13.6273579625 , 22.045833699200003 , 42.47373813583333 , 36.216295980000005 , 14.992245947142857 , 117.66265314968261 , 41.946033608695636 , 47.924031835609746 , 52.03324387033333 , 214.956521425 , 22.547546988750003 , 17.5561776467742 , 16.851157194 , 13.733851305000002 , 28.688326671555558 , 18.18781114357143 , 9.356771754999999 ,

cover : 

    21008.60380710909 , 18737.927849844167 , 7953.791551000001 , 28094.847689434995 , 28640.625635751025 , 20318.014958142852 , 11079.217108192692 , 20793.64533 , 15219.778178192175 , 5943.57649032 , 15093.768533909093 , 14667.623076329375 , 32799.39516666666 , 9381.15051525 , 27849.383293 , 25787.975759166668 , 3333.79367 , 9120.701479233334 , 3061.3389820633333 , 11081.130386 , 17848.629699832 , 27976.600110587515 , 11796.579860526923 , 31279.633359571428 , 21269.423612692062 , 19920.45543073913 , 26620.894694024395 , 10580.368789466667 , 38514.38807374999 , 21029.157458416666 , 29838.309997580647 , 25030.11496066667 , 17080.157988811112 , 27460.434160822228 , 16954.942480214286 , 23489.0688743 ,

total_gain : 

    1670.0039639549998 , 1852.5033133499999 , 197.57875743000002 , 1402.7383291999997 , 1792.0889143699997 , 1530.1075189199998 , 442.74430108000007 , 449.7003485899999 , 367.93457963 , 96.56940602 , 618.79057499 , 389.90717643 , 64.6665191 , 55.442563899999996 , 1284.7575769900002 , 119.68167357000002 , 53.63121895 , 152.89320176999996 , 53.51740695 , 54.50943185 , 551.14584248 , 1019.3697152599999 , 941.6236954800002 , 104.94572163 , 7412.747148430005 , 964.7587729999997 , 1964.8853052599995 , 312.19946322199996 , 1719.6521714 , 541.1411277300001 , 544.2415070500001 , 252.76735791 , 247.20932349000003 , 1290.9747002200002 , 254.62935601000004 , 93.56771754999998 ,

total_cover : 

    462189.28375640005 , 449710.26839626 , 95445.49861200001 , 561896.9537886999 , 1403390.6561518002 , 568904.4188279998 , 288059.64481301 , 291111.03462 , 350054.89809842 , 59435.7649032 , 166031.45387300002 , 234681.96922127 , 98398.18549999999 , 37524.602061 , 640535.815739 , 154727.854555 , 13335.17468 , 109448.4177508 , 18368.03389238 , 44324.521544 , 446215.7424958 , 671438.4026541003 , 306711.0763737 , 218957.433517 , 1339973.6875995998 , 458170.474907 , 1091456.6824550002 , 63482.2127368 , 308115.10458999994 , 504699.779002 , 924987.6099250001 , 375451.72441 , 307442.8437986 , 1235719.5372370002 , 237369.194723 , 234890.68874299998 ,


#### 4.3.2.3 residents_feature_subset_perf4g
##### Logistic Regression

Residents LR 10CV f1_weighted scores : 

    [0.50298731 0.50202774 0.49625421 0.49986265 0.49991583 0.50502329 0.50601378 0.51098394 0.50914627 0.5064319 ]

Residents LR classification report :

              precision    recall  f1-score   support
           0       0.60      0.93      0.73     13497
           1       0.53      0.10      0.17      9349
       micro avg   0.59      0.59      0.59     22846
       macro avg   0.56      0.52      0.45     22846
    weighted avg   0.57      0.59      0.50     22846

Residents LR confusion matrix :

    [[12619   878]
     [ 8376   973]]

##### SVM
Residents SVM 10CV f1_weighted scores : 

    [0.50093968 0.50272392 0.49861236 0.50027948 0.4999116  0.5036059 0.50556904 0.50682809 0.50933262 0.50744984]

Residents SVM classification report :

              precision    recall  f1-score   support
           0       0.60      0.94      0.73     13497
           1       0.53      0.10      0.17      9349
       micro avg   0.60      0.60      0.60     22846
       macro avg   0.56      0.52      0.45     22846
    weighted avg   0.57      0.60      0.50     22846

Residents SVM confusion matrix :

    [[12638   859]
     [ 8390   959]]

##### XGBOOST

Residents XGBOOST 10CV f1_weighted scores : 

    [0.54016209 0.54406173 0.53684772 0.54063281 0.55200052 0.55015609 0.54751184 0.55345481 0.55266474 0.5479062 ]
 
Residents XGBOOST classification report :

              precision    recall  f1-score   support
           0       0.61      0.89      0.73     13497
           1       0.54      0.19      0.28      9349
       micro avg   0.60      0.60      0.60     22846
       macro avg   0.58      0.54      0.50     22846
    weighted avg   0.58      0.60      0.54     22846

Residents XGBOOST confusion matrix :

    [[11989  1508]
     [ 7571  1778]]
 
Residents XGBOOST features importances :

    [0.05896477 0.01808032 0.01340265 0.00964418 0.02157999 0.08890527
     0.01617851 0.12717989 0.03185442 0.01596039 0.00983383 0.01095722
     0.00955852 0.01295907 0.01203345 0.01627942 0.01051013 0.00937814
     0.00952983 0.02251766 0.02049096 0.02034708 0.01825706 0.01113681
     0.12280573 0.0368738  0.01731363 0.01205648 0.02216321 0.01168265
     0.02250974 0.01173889 0.07049224 0.04776671 0.01536269 0.01369467]

weight : 

    32 , 22 , 15 , 14 , 30 , 52 , 32 , 11 , 48 , 1 , 12 , 12 , 4 , 6 , 19 , 7 , 5 , 3 , 5 , 13 , 23 , 18 , 15 , 2 , 90 , 5 , 17 , 20 , 16 , 12 , 26 , 9 , 25 , 62 , 8 , 8 ,

gain : 
    
    92.65803978749997 , 28.411662968181815 , 21.061106107999997 , 15.154999392857144 , 33.91109858333334 , 139.7069503578846 , 25.423140282500004 , 199.85222465454552 , 50.05646505208333 , 25.0803757 , 15.453017209999999 , 17.21832593333333 , 15.020389825 , 20.3640623 , 18.909524962631576 , 25.581707285714288 , 16.5157629 , 14.7369299 , 14.975306708 , 35.38456726153846 , 32.19977627 , 31.97368016111111 , 28.689398905999997 , 17.500540700000002 , 192.9786009544444 , 57.94399432 , 27.206874115294124 , 18.9457195675 , 34.827581165000005 , 18.3582706525 , 35.37210275384615 , 18.446654216 , 110.77246858760003 , 75.06126641145161 , 24.14114117 , 21.51998517375 ,

cover : 

    26387.519673 , 21180.354544090907 , 18145.578023206665 , 14246.851055692856 , 20747.686722660004 , 27427.16037801923 , 12052.327777378123 , 17120.122973636364 , 22604.380157810418 , 1830.65332 , 10520.562578665 , 13988.975095583335 , 21799.734774499997 , 6298.978348333333 , 24889.30388963158 , 19667.415624285717 , 8224.7098504 , 4099.947183333334 , 37114.649913999994 , 30240.29437307692 , 23872.73195608696 , 17329.477162777777 , 24133.909383735336 , 12276.068720000001 , 25100.899892464447 , 28470.881536 , 20622.566630970003 , 8963.235991146501 , 20991.957608999997 , 16072.366388891665 , 24626.526438538458 , 5873.348169144445 , 17491.723564204 , 24193.491513677418 , 4190.419806225 , 12670.4562699375 ,

total_gain : 

    2965.057273199999 , 625.0565852999999 , 315.91659161999996 , 212.1699915 , 1017.3329575000002 , 7264.76141861 , 813.5404890400001 , 2198.3744712000007 , 2402.7103225 , 25.0803757 , 185.43620651999998 , 206.61991119999996 , 60.0815593 , 122.1843738 , 359.28097428999996 , 179.071951 , 82.5788145 , 44.2107897 , 74.87653354 , 459.99937439999997 , 740.59485421 , 575.5262428999999 , 430.34098358999995 , 35.001081400000004 , 17368.074085899996 , 289.7199716 , 462.5168599600001 , 378.91439135 , 557.2412986400001 , 220.29924783 , 919.6746715999999 , 166.019887944 , 2769.311714690001 , 4653.79851751 , 193.12912936 , 172.15988139 ,

total_cover : 

    844400.629536 , 465967.79997 , 272183.67034809996 , 199455.9147797 , 622430.6016798001 , 1426212.339657 , 385674.48887609993 , 188321.35271 , 1085010.2475749 , 1830.65332 , 126246.75094398 , 167867.701147 , 87198.93909799999 , 37793.87009 , 472896.77390300005 , 137671.90937 , 41123.549252000004 , 12299.841550000001 , 185573.24956999999 , 393123.82684999995 , 549072.83499 , 311930.58892999997 , 362008.64075603004 , 24552.137440000002 , 2259080.9903218 , 142354.40768 , 350583.63272649003 , 179264.71982293003 , 335871.32174399996 , 192868.3966667 , 640289.687402 , 52860.133522300006 , 437293.0891051 , 1499996.473848 , 33523.3584498 , 101363.6501595 ,
