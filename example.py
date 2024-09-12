import pandas as pd

from soccerlib import DataLoader as DL
from soccerlib import DataProcessing as DP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings

# Отключаем все предупреждения
warnings.filterwarnings('ignore')

dic_train = {}
#years = ['2021-2022', '2022-2023']
#years = ['2022-2023', '2020-2021', '2018-2019','2024-2025']
years = ['2022','2026']
#chmps = [12, 13, 15, 16, 17, 18, 19, 20, 2032]
#chmps = [1658, 1659, 1660, 1661]
chmps = [741]
for year in years:
    dic_train[year] = []
    for chmp in chmps:
        dic_train[year].append(f'data\dataset_{chmp}_{year}.csv')


# years = ['2023']
# chmps = [613]
# for year in years:
#     dic_train[year] = []
#     for chmp in chmps:
#         dic_train[year].append(f'data\dataset_{chmp}_{year}.csv')


list_test = []
#years = ['2023-2024']
#years = ['2024-2025']
years = ['2026']
#chmps = [12, 13, 15, 16, 17, 18, 19, 20, 2032]
#chmps = [737]
for year in years:
    for chmp in chmps:
        list_test.append(f'data\dataset_{chmp}_{year}.csv')

#list_predict = [f'data\dataset_predictions.csv']
list_predict = [f'data\dataset_predictions_south_america.csv']


session = DL.DataLoader()
df_train = session.fit(dic_train)
df_test = session.transform(list_test)
df_predict = session.transform_predict(list_predict)

# Найти индекс строки с максимальным значением в столбце 'rating_home'
max_index = df_train['rating_guest'].idxmax()

# Вывести строку с этим индексом
max_row = df_train.loc[max_index]
print(max_row)

session = DP.DataProcessing(
                        X_columns=('stat_params',
                                   'name_teams',
                                   'ratings'),
                        columns_to_drop=('high_corr',),
                        encodings=('teams_binary',),
                        y_columns=['result'],
                        imputer='iter')

X_train, y_train = session.fit_transform(df_train)
X_test, y_test = session.transform(df_test)
X_predict = session.transform_predict(df_predict)

#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = LogisticRegression(max_iter=1000)  # Увеличиваем max_iter для лучшей сходимости
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_predict = model.predict(X_predict)
# Вычисление точности для тренировочного набора
accuracy = accuracy_score(y_train, y_pred_train)
print('Train Accuracy:', accuracy)

accuracy = accuracy_score(y_test, y_pred_test)
print('Test Accuracy:', accuracy)

df_predict['pred_res'] = y_pred_predict
print(df_predict[['team_home','team_guest','pred_res']])
#print(pd.concat([df_predict[['team_home','team_guest','pred_res']], y_test],axis=1))

