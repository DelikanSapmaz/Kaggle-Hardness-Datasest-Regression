import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import math
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate,validation_curve
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler,MinMaxScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error



pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)

df = pd.read_csv("kaggle_yarisma_new/Regression with a Mohs Hardness Dataset/train.csv")

df_train = df.copy()

df_test = pd.read_csv("kaggle_yarisma_new/Regression with a Mohs Hardness Dataset/test.csv")

df_test_copy = df_test.copy()

ids = df_test_copy["id"]

def check_data(dataframe):
    print("################ HEAD ################")
    print(dataframe.head())
    print("################ TAIL ################")
    print(dataframe.tail())
    print("################ DESCRİBE ################")
    print(dataframe.describe().T)
    print("################ SHAPE ################")
    print(dataframe.shape)
    print("################ INFO ################")
    print(dataframe.info())
    print("################ ISNULL ################")
    print(dataframe.isnull().sum())


check_data(df_train)

def grab_cols_names(dataframe,cat_th=10,car_th=20):
    """

    Parameters
    ----------
    dataframe
    cat_th
    car_th

    Returns
    -------
    cat_cols,num_cols,cat_but_car
    """
    # categorical columns

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes not in ["int64","float64"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64","float64"]
                   and dataframe[col].nunique()<cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes  not in ["int64","float64"]
                   and dataframe[col].nunique()>car_th]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerical columns

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64","float64"]]

    num_cols = [col for col in num_cols if col not in cat_cols]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')


    return cat_cols,num_cols,cat_but_car




cat_cols,num_cols,cat_but_car = grab_cols_names(df_train)

num_cols = [col for col in num_cols if col not in ["Hardness","id"]]

def num_cols_analys(dataframe,num_cols,plot=False):
    qunatile = [0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.99]
    print(dataframe[num_cols].describe(qunatile).T)

    if plot:
        fig = plt.figure(figsize=(20,12))
        ax = fig.gca()
        dataframe[num_cols].hist(ax=ax)
        plt.show(block=True)

num_cols_analys(df_train,num_cols,plot=True)

# Correlation Matrix
f,ax = plt.subplots(figsize = [20,14])
sns.heatmap(df_train[num_cols].corr(),annot=True,fmt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


# Base Model Holdout Method

X =df_train.drop("Hardness",axis=1)
y = df_train["Hardness"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

## Model 1: xgboost
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train,y_train)

y_pred_xgb = xgb_reg.predict(X_test)

mean_squared_error(y_test,y_pred_xgb)
# mean_squared_error 1.61
# root mean_squared_error
math.sqrt(mean_absolute_error(y_test,y_pred_xgb))
# root mean_squared_error 0.97

mean_absolute_error(y_test,y_pred_xgb)
# mean_absolute_error 0.94

# Base Model Crosvalidation

models = [("XGB", XGBRegressor(random_state=1234)),
         ("CatBoostRegressor", CatBoostRegressor(verbose=False,random_state=1234)),
         ("RandomForestRegressor", RandomForestRegressor(random_state=1234)),
         ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=1234)),
         ("LGBMRegressor", LGBMRegressor(random_state=1234))]


# models listesini tanımladığınız yer

for name, model in models:
    cv_result = cross_validate(model, X, y, cv=10, scoring=make_scorer(mean_squared_error, greater_is_better=False))
    print(f"########### {name} ###########")
    print(f"mean_squared_error: {round(-cv_result['test_score'].mean(), 4)}")

# CatBoostRegressor mean_squared_error: 1.4899
# LGBMRegressor mean_squared_error: 1.4781

# Outliers


def outliers_threshold(dataframe,num_cols):
    quartile3 = dataframe[num_cols].quantile(0.95)
    quartile1 = dataframe[num_cols].quantile(0.05)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5*interquantile_range
    low_limit =  quartile1 - 1.5*interquantile_range
    return up_limit,low_limit


def check_outliers(dataframe,num_cols):
    up,low = outliers_threshold(dataframe,num_cols)

    if dataframe.loc[(dataframe[num_cols]>up) | (dataframe[num_cols]<low)].any(axis=None):
        return num_cols,True
    else:
        return num_cols,False


for col in num_cols:
    print(check_outliers(df_train,col))


def replace_with_threshold(dataframe,num_cols):
    up,low = outliers_threshold(dataframe,num_cols)
    dataframe.loc[(dataframe[num_cols]>up),num_cols] = up
    dataframe.loc[(dataframe[num_cols]<low),num_cols] = low


for col in num_cols:
    replace_with_threshold(df_train,col)

# Feature Importance

xgb_reg.feature_importances_
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')




plot_importance(xgb_reg,X)

# Feature Engineering

df_train['Allelectrons_val_e_Ratio'] = df_train['allelectrons_Average'] / df_train['val_e_Average']

df_train['Density_per_atomicweight'] = df_train['density_Average'] / df_train['atomicweight_Average']

df_train['El_neg_ionenergy_Relation'] = df_train['el_neg_chi_Average'] * df_train['ionenergy_Average']

df_train['R_vdw_cov_Difference'] = df_train['R_vdw_element_Average'] - df_train['R_cov_element_Average']

df_train['Elec_struct_Combination'] = df_train['allelectrons_Average'] * df_train['R_cov_element_Average']

df_train['Zaratio_density_Relation'] = df_train['zaratio_Average'] * df_train['density_Average']

cat_cols,num_cols,cat_but_car=grab_cols_names(df_train)

num_cols = [col for col in num_cols if col not in "Hardness"]

df_train.isnull().sum()
# Density_per_atomicweight  14 Null

df_train = df_train.replace([np.inf, -np.inf], np.nan)# we filled infinite values in our data with empty values
# verimizdeki sonsuz değerleri boşdeğerler ile dolduruk

# KNN ımputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_train = pd.DataFrame(imputer.fit_transform(df_train),columns=df_train.columns)

df_train = df_train.apply(lambda x:x.fillna(x.mean()) if x.dtype!="O" else x ,axis=0)

# Scaling

cat_cols,num_cols,cat_but_car=grab_cols_names(df_train)

num_cols = [col for col in num_cols if col not in ["Hardness","id"]]

scaler = MinMaxScaler()

df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

df_train.drop("id",axis=1,inplace=True)

#                                           Final Model
# CatBoostRegressor mean_squared_error: 1.4899
# LGBMRegressor mean_squared_error: 1.4781

X =df_train.drop("Hardness",axis=1)
y = df_train["Hardness"]


models = [("XGB", XGBRegressor(random_state=1234)),
         ("CatBoostRegressor", CatBoostRegressor(verbose=False,random_state=1234)),
         ("RandomForestRegressor", RandomForestRegressor(random_state=1234)),
         ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=1234)),
         ("LGBMRegressor", LGBMRegressor(random_state=1234))]



for name, model in models:
    cv_result = cross_validate(model, X, y, cv=10, scoring=make_scorer(mean_squared_error, greater_is_better=False))
    print(f"########### {name} ###########")
    print(f"mean_squared_error: {round(-cv_result['test_score'].mean(), 4)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)


xgb_reg = XGBRegressor()
xgb_reg.get_params()

param_list = {"max_depth": range(2, 10), "subsample": [0.2, 0.4, 0.6, 0.8, 1.0]}

grd = GridSearchCV(estimator=xgb_reg,
                   cv=5,
                   n_jobs=-1,
                   param_grid=param_list).fit(X_train,y_train)


grd.best_params_

xgb_reg.set_params(**grd.best_params_).fit(X_train,y_train)


y_pred_xgb2 = xgb_reg.predict(X_test)

mean_squared_error(y_test,y_pred_xgb)# 1.6132832662653007,



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')




plot_importance(xgb_reg,X)


# Model 2 CatBoostRegressor
param_dist = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.001, 0.005, 0.01, 0.1],
    'depth': [2, 3, 4, 5, 6],
}

catboost_model = CatBoostRegressor()

grd_catboost = GridSearchCV(
    estimator=catboost_model,
    cv=5,
    n_jobs=-1,
    param_grid=param_dist
).fit(X_train, y_train)



grd_catboost.best_params_


catboost_model.set_params(**grd_catboost.best_params_).fit(X_train,y_train)

y_pred_catboost=catboost_model.predict(X_test)


mean_squared_error(y_test,y_pred_xgb)
#  1.6132832662653007


# LGBMRegressor Model


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 3, 4, 5, 6],
    'num_leaves': [20, 30, 40, 50],
}

lgbm_model = LGBMRegressor()

grd_lgbm = GridSearchCV(
    estimator=lgbm_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
).fit(X_train,y_train)


grd_lgbm.best_params_

lgbm_model.set_params(**grd_lgbm.best_params_).fit(X_train,y_train)

y_pred_lgbm = lgbm_model.predict(X_test)

mean_squared_error(y_test,y_pred_lgbm)
# mean_squared_error  1.4900024593394277

#                                       Test Data



df_test_copy['Allelectrons_val_e_Ratio'] = df_test_copy['allelectrons_Average'] / df_test_copy['val_e_Average']

df_test_copy['Density_per_atomicweight'] = df_test_copy['density_Average'] / df_test_copy['atomicweight_Average']

df_test_copy['El_neg_ionenergy_Relation'] = df_test_copy['el_neg_chi_Average'] * df_test_copy['ionenergy_Average']

df_test_copy['R_vdw_cov_Difference'] = df_test_copy['R_vdw_element_Average'] - df_test_copy['R_cov_element_Average']

df_test_copy['Elec_struct_Combination'] = df_test_copy['allelectrons_Average'] * df_test_copy['R_cov_element_Average']

df_test_copy['Zaratio_density_Relation'] = df_test_copy['zaratio_Average'] * df_test_copy['density_Average']

cat_cols,num_cols,cat_but_car=grab_cols_names(df_test_copy)


df_test_copy.isnull().sum()
# Density_per_atomicweight  14 Null

df_test_copy = df_test_copy.replace([np.inf, -np.inf], np.nan)# we filled infinite values in our data with empty values
# verimizdeki sonsuz değerleri boşdeğerler ile dolduruk




# KNN ımputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_test_copy = pd.DataFrame(imputer.fit_transform(df_test_copy),columns=df_test_copy.columns)

# df_test_copy = df_test_copy.apply(lambda x:x.fillna(x.mean()) if x.dtype!="O" else x ,axis=0)

# Scaling

cat_cols,num_cols,cat_but_car=grab_cols_names(df_test_copy)

num_cols = [col for col in num_cols if col not in "id"]

scaler = MinMaxScaler()

df_test_copy[num_cols] = scaler.fit_transform(df_test_copy[num_cols])

df_test_copy.drop("id",axis=1,inplace=True)

y_pred_test_data=lgbm_model.predict(df_test_copy)


data = pd.DataFrame({"id":ids,
                     "Hardness":y_pred_test_data})

data.to_csv("kaggle_yarisma_new/Regression with a Mohs Hardness Dataset/Hardness.csv",index=False)



