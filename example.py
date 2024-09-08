from soccerlib import DataLoader as DL
from soccerlib import DataProcessing as DP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dic_train = {}
years = ['2021-2022', '2022-2023']
chmps = [12, 13, 15, 16, 17, 18, 19, 20, 2032]
for year in years:
    dic_train[year] = []
    for chmp in chmps:
        dic_train[year].append(f'data\dataset_{chmp}_{year}.csv')

list_test = []
years = ['2023-2024']
chmps = [12, 13, 15, 16, 17, 18, 19, 20, 2032]
for year in years:
    for chmp in chmps:
        list_test.append(f'data\dataset_{chmp}_{year}.csv')



session = DL.DataLoader()
df_train = session.fit(dic_train)
df_test = session.transform(list_test)

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
                        y_columns=['result'],
                        imputer='iter')

X_train, y_train = session.fit_transform(df_train)
X_test, y_test = session.transform(df_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
# Вычисление точности для тренировочного набора
accuracy = accuracy_score(y_train, y_pred_train)
print('Train Accuracy:', accuracy)

accuracy = accuracy_score(y_test, y_pred_test)
print('Test Accuracy:', accuracy)
