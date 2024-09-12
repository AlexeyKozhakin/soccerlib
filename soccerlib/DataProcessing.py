import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from soccerlib.DataCleaningFunctions import (clean_goal_time,
                                   load_stat_params,
                                   get_final_columns,
                                   find_highly_correlated_pairs,
                                   question_sign_tire_to_nan)

class DataProcessing:
    def __init__(self,
                        X_columns=(
                        'stat_params',
                        'label_teams',
                        'ratings'),
                        encodings=(
                        'teams_binary',
                        ),
                        columns_to_drop=('high_corr',),
                        y_columns=('Время гола',),
                        imputer='iter'):
        self.X_columns = X_columns
        self.y_columns = y_columns
        self.columns_to_drop = []
        self.columns_to_keep = []
        self.imputer_type = imputer
        self.encodings = encodings

    def fit_transform(self, raw_data):
        #================ Убрать косяки в результатах ==============
        #================ Binary encodings =========================
        name_teams = []
        if 'teams_binary' in self.encodings:
            self.teams_home_encoder = ce.BinaryEncoder(cols=['team_home'])
            data_teams_home_encoder = self.teams_home_encoder.fit_transform(raw_data['team_home'])
            self.teams_guest_encoder = ce.BinaryEncoder(cols=['team_guest'])
            data_teams_guest_encoder = self.teams_guest_encoder.fit_transform(raw_data['team_guest'])
            raw_data = pd.concat([raw_data,
                                  data_teams_home_encoder,
                                  data_teams_guest_encoder
                                  ],
                                 axis=1)
            name_teams = list(data_teams_home_encoder.columns) + list(data_teams_guest_encoder.columns)
            print(f'Новые колонки home:{name_teams}')
            print(f'home:{len(data_teams_home_encoder.columns)}')
            #print(f'guest:{len(data_teams_guest_encoder.columns)}')
        #============ Добавление названия команды ====================
        elif 'name_teams' in self.X_columns:
            name_teams = ['team_index_home', 'team_index_guest']
        #=========================== Очистка от 45+, 90+
        if 'Время гола' in self.y_columns:
            raw_data = clean_goal_time(raw_data)
            raw_data = raw_data[pd.notna(raw_data['Время гола'])]
        #Шаг 1 - финальный список столбцов, которые оставляем
        stat_params = []
        if 'stat_params' in self.X_columns:
            stat_params = load_stat_params()

        # ============ Добавление рейтинга команды ===================
        rating_teams = []
        if 'ratings' in self.X_columns:
            rating_teams = ['rating_home', 'rating_guest']
        init_columns = raw_data.columns
        columns_to_keep = stat_params
        columns_to_drop = []
        final_columns = get_final_columns(init_columns, columns_to_keep, columns_to_drop)
        #Шаг 2 - убираем лишние столбцы

        df = raw_data[final_columns].copy()
        # Очистка данных - удаление строк с пропущенными значениями ('?' и '-')
        df = question_sign_tire_to_nan(df)
        df.dropna(inplace=True)
        p1, p2 = find_highly_correlated_pairs(df, threshold=0.85)
        high_corr = []
        if 'high_corr' in columns_to_drop:
            high_corr = p1
        #================= Split X, y =================================
        # Выбираем все столбцы, кроме целевого параметра (время первого гола)
        init_columns = raw_data.columns
        columns_to_keep = stat_params + name_teams + rating_teams
        columns_to_drop = high_corr
        self.final_columns = get_final_columns(init_columns, columns_to_keep, columns_to_drop)
        print(columns_to_keep)
        # Очистка данных - удаление строк с пропущенными значениями ('?' и '-')

        X = raw_data[self.final_columns].copy()
        X = question_sign_tire_to_nan(X)
        # Целевой параметр - время первого гола
        y = raw_data[self.y_columns]

        # Проверим разделение
        print(f"X (входные параметры): {X.shape}")
        print(f"y (прогнозируемый параметр): {y.shape}")
        # ================ Fill NAN ===================================
        if self.imputer_type=='iter':
            self.imputer = IterativeImputer(random_state=100, max_iter=10)
            # fit on the dataset
            self.imputer.fit(X)
            X_iter = self.imputer.transform(X)
        elif self.imputer_type=='knn':
            pass
        return X_iter, y

    def transform(self, raw_test_data):

        if 'teams_binary' in self.encodings:
            data_teams_home_encoder = self.teams_home_encoder.transform(raw_test_data['team_home'])
            data_teams_guest_encoder = self.teams_guest_encoder.transform(raw_test_data['team_guest'])
            raw_test_data = pd.concat([raw_test_data,
                                  data_teams_home_encoder,
                                  data_teams_guest_encoder
                                  ],
                                 axis=1)

        X = raw_test_data[self.final_columns].copy()
        X = question_sign_tire_to_nan(X)
        # Целевой параметр - время первого гола
        y = raw_test_data[self.y_columns]

        # Проверим разделение
        print(f"X (входные параметры): {X.shape}")
        print(f"y (прогнозируемый параметр): {y.shape}")
        # ================ Fill NAN ===================================
        self.imputer.fit(X)
        X_iter = self.imputer.transform(X)
        return X_iter, y

    def transform_predict(self, raw_test_data):

        if 'teams_binary' in self.encodings:
            data_teams_home_encoder = self.teams_home_encoder.transform(raw_test_data['team_home'])
            data_teams_guest_encoder = self.teams_guest_encoder.transform(raw_test_data['team_guest'])
            raw_test_data = pd.concat([raw_test_data,
                                  data_teams_home_encoder,
                                  data_teams_guest_encoder
                                  ],
                                 axis=1)

        X = raw_test_data[self.final_columns].copy()
        X = question_sign_tire_to_nan(X)

        # Проверим разделение
        print(f"X (входные параметры): {X.shape}")
        # ================ Fill NAN ===================================
        self.imputer.fit(X)
        X_iter = self.imputer.transform(X)
        return X_iter
