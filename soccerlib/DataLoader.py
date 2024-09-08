import pandas as pd
from soccerlib.DataCleaningFunctions import (get_unique_teams,
                                             add_rating_to_dataset_fit,
                                             add_rating_to_dataset_transform,
                                             add_team_label_to_dataset)

def DataLoader_test(file_paths):
    datasets_list = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(datasets_list, ignore_index=True)
    return data

def DataLoader_train(dict_train_files):
    datasets_lists_years = []
    for file_paths in dict_train_files.values():
        datasets_list = [pd.read_csv(file) for file in file_paths]
        data = pd.concat(datasets_list, ignore_index=True)
        datasets_lists_years.append(data)
    #====== Если хотим добавлять =========================
    teams = get_unique_teams(datasets_lists_years)
    team_index = {team: i for i, team in enumerate(teams)}
    data_all_years = pd.concat(datasets_lists_years, ignore_index=True)
    data_all_years, dict_rating = add_rating_to_dataset_fit(data_all_years)
    data_all_years  = add_team_label_to_dataset(data_all_years, team_index)
    print(data_all_years)
    print(teams)
    print(len(teams))
    return data_all_years, dict_rating, team_index

class DataLoader:

    def __init__(self, rating=False, teams_label=False):
        self.rating = rating
        self.teams_label = teams_label

    def fit(self, dict_train_files):
        data_all_years, self.dict_rating, self.team_index = DataLoader_train(dict_train_files)

        return data_all_years

    def transform(self, list_test_files):
        data_test = DataLoader_test(list_test_files)

        data_test = add_rating_to_dataset_transform(data_test,
                                                    self.dict_rating)
        data_test  = add_team_label_to_dataset(data_test, self.team_index)
        return data_test






