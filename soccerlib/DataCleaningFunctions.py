import requests
from bs4 import BeautifulSoup as BS
import os
import numpy as np
import pandas as pd



def add_team_label_to_dataset(df_in, team_index):
    df = df_in.copy()
    # Применение индексов к столбцу с командами
    df['team_index_home'] = df['team_home'].map(team_index)
    # Замена NaN на -1 для команд, которых нет в словаре
    df['team_index_home'] = df['team_index_home'].fillna(-1).astype(int)
    # Применение индексов к столбцу с командами
    df['team_index_guest'] = df['team_guest'].map(team_index)
    # Замена NaN на -1 для команд, которых нет в словаре
    df['team_index_guest'] = df['team_index_guest'].fillna(-1).astype(int)
    return df

def add_rating_to_dataset_fit(df_in):
    df = df_in.copy()
    team_ratings = get_rating(df)
    rating_dict = dict(zip(team_ratings['team'], team_ratings['rating']))
    # Добавляем новый столбец с рейтингами в DataFrame matches
    df['rating_home'] = df['team_home'].map(rating_dict)
    df['rating_guest'] = df['team_guest'].map(rating_dict)
    return df, rating_dict

def add_rating_to_dataset_transform(df_in, rating_dict):
    df = df_in.copy()
    # Добавляем новый столбец с рейтингами в DataFrame matches
    df['rating_home'] = df['team_home'].map(rating_dict)
    df['rating_guest'] = df['team_guest'].map(rating_dict)
    return df

def get_rating(df_in):
    df = df_in.copy()
    # Найти максимальное значение по двум столбцам
    # Преобразовать тип данных столбцов 'sc_h' и 'sc_g' в int
    # Преобразовать столбцы 'sc_h' и 'sc_g' в числовой тип, заменив нечисловые значения на NaN
    df['sc_h'] = pd.to_numeric(df['sc_h'], errors='coerce')
    df['sc_g'] = pd.to_numeric(df['sc_g'], errors='coerce')
    print('shape df:', df.shape)
    # Удалить строки, где в 'sc_h' или 'sc_g' есть NaN (то есть там, где были нечисловые значения)
    df = df.dropna()
    ngoal_max = df[['sc_h', 'sc_g']].max().max() + 1
    #ngoal_max = 20
    print(type(ngoal_max))
    print(ngoal_max)
    # Получаем уникальные команды
    teams = list(set(df['team_home']).union(set(df['team_guest'])))
    n = len(teams)
    print('n-типа сколько команд:',n)
    # Создаем словарь индексов команд
    team_idx = {team: i for i, team in enumerate(teams)}
    # Матрица коэффициентов и вектор свободных членов
    print('max_team_idx:', max(team_idx.values()))
    A = np.zeros((2 * len(df), n))
    b = np.zeros(2 * len(df))
    print('shape A:',A.shape)
    print('shape df:', df.shape)
    # Заполняем систему уравнений
    for i in range(df.shape[0]):
        team1 = df['team_home'].iloc[i]
        team2 = df['team_guest'].iloc[i]
        score1 = df['sc_h'].iloc[i]
        score2 = df['sc_g'].iloc[i]

        idx1 = team_idx[team1]
        idx2 = team_idx[team2]

        # Первое уравнение: R_i - R_j + b1 = score1

        A[2 * i, idx1] = -1
        A[2 * i, idx2] = 1
        # print(score1)
        # print(type(score1))
        b[2 * i] = np.log(ngoal_max/(score1+0.01)-1)

        # Второе уравнение: R_j - R_i + b2 = score2
        A[2 * i + 1, idx1] = 1
        A[2 * i + 1, idx2] = -1
        b[2 * i + 1] = np.log(ngoal_max/(score2+0.01)-1)
    # Решаем систему уравнений
    R = np.linalg.lstsq(A, b, rcond=None)[0]
    # Выводим результаты
    team_ratings = {'team':[],'rating':[]}
    for team, idx in team_idx.items():
      team_ratings['team'].append(team)
      team_ratings['rating'].append(R[idx])

    return team_ratings


def get_unique_teams(df_list):
    # Используем list comprehension для создания списка уникальных команд для каждого датафрейма
    team_sets = [set(np.unique(df[['team_home', 'team_guest']].values)) for df in df_list]
    print('Общее число команд:',len(set.union(*team_sets)))
    # Найдем пересечение всех множеств команд
    unique_teams = set.intersection(*team_sets)
    return unique_teams

def clean_goal_time(df):
    def time_extra(time_goal_row):
        n = time_goal_row.find('+')
        if n != -1:
            return time_goal_row[:n]  # Возвращает часть строки до символа '+'
        return time_goal_row  # Если '+' нет, возвращается исходное значение
    # Применение функции ко всему столбцу "Время гола"
    df['Время гола'] = df['Время гола'].apply(time_extra)
    return df

def get_final_columns(init_columns, columns_to_keep=[], columns_to_drop=[]):
    # Проверка наличия всех столбцов из columns_to_keep в init_columns
    missing_keep = [col for col in columns_to_keep if col not in init_columns]
    if missing_keep:
        print(f"Недостающие столбцы для сохранения: {', '.join(missing_keep)}")

    # Проверка наличия всех столбцов из columns_to_drop в init_columns
    missing_drop = [col for col in columns_to_drop if col not in init_columns]
    if missing_drop:
        print(f"Недостающие столбцы для удаления: {', '.join(missing_drop)}")

    # Формирование финального списка столбцов
    final_columns = [col for col in columns_to_keep if col in init_columns and col not in columns_to_drop]

    return final_columns

def get_stat_params_from_page(soup):
    column_names = []
    tables = soup.find_all(class_="tablesorter")
    soup = tables[0]
    # Найдем все строки в теле таблицы
    rows = soup.find('tbody').find_all('tr')
    for row in rows:
      cells = row.find_all('td')
      # Extract text from the cells
      name_param = cells[1].get_text(strip=True)
      exclude_param = ['Матчи']
      if name_param not in exclude_param:
         column_names += [name_param+'-home', name_param+'-guest']
    # берем данные из таблицы 2
    soup = tables[1]
    # Найдем все строки в теле таблицы
    rows = soup.find('tbody').find_all('tr')
    for row in rows:
      cells = row.find_all('td')
      # Extract text from the cells
      name_param = cells[1].get_text(strip=True)
      if name_param not in exclude_param:
         column_names += [name_param+'-home', name_param+'-guest']
    return column_names

def save_stat_parameters_names(soup, file_name='stat_column_names.txt'):
    # Берем названия столбцов
    column_names = get_stat_params_from_page(soup)
    # Сохранение в файл TXT
    # Открытие файла в режиме записи
    with open(file_name, "w", encoding="utf-8") as file:
        # Запись каждого слова в новую строку файла
        for name in column_names:
            file.write(name + "\n")

def read_column_names_stat_from_file(file_name='stat_column_names.txt'):
    # Открытие файла в режиме чтения
    with open(file_name, "r", encoding='utf-8') as file:
        # Чтение всех строк из файла и удаление символов новой строки
        loaded_column_names = [line.strip() for line in file]
        return loaded_column_names

def question_sign_tire_to_nan(df):
    df_out = df.copy()
    df_out.replace(['?', '-'], np.nan, inplace=True)
    df_out = df_out.apply(pd.to_numeric)
    return df_out

def load_stat_params(
    file_name='stat_column_names.txt',
    load_page = f'https://soccer365.ru/games/1903293/&tab=form_teams'):
    # Проверяем, существует ли файл в текущей директории
    if os.path.isfile(file_name):
        print(f"Файл '{file_name}' существует. Столбцы будут загружены из него")
    else:
        print(f"Файл '{file_name}' не найден.\n"\
        f"Будет произведена загрузка со строницы {load_page}")
        html_games = requests.get(load_page)
        soup = BS(html_games.text, "html.parser")
        save_stat_parameters_names(soup, file_name=file_name)
    stat_parameters = read_column_names_stat_from_file(file_name=file_name)
    return stat_parameters

def find_highly_correlated_pairs(df, threshold=0.85):
    # Рассчет корреляции Пирсона между всеми параметрами
    correlation_matrix = df.corr()

    # Списки для хранения сильно скоррелированных пар
    param1 = []
    param2 = []

    # Поиск пар параметров с коэффициентом корреляции выше заданного порога
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                param1.append(correlation_matrix.columns[i])
                param2.append(correlation_matrix.columns[j])

    return param1, param2

def create_result_column(df):
    # df['result'] = create_result_column(df)
    # Условие и соответствующие значения
    conditions = [
        df['sc_h'] > df['sc_g'],
        df['sc_h'] == df['sc_g'],
        df['sc_h'] < df['sc_g']
    ]
    values = [1, 0, 2]
    # Создание нового столбца
    return np.select(conditions, values)

