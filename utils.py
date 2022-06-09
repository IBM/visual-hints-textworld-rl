import re
import glob
import os

__ALL_ACTIONS = {'cook', 'cut', 'drop', 'go', 'open', 'take'}
def get_files(root_dir, allowed_actions, kind='train'):
    def filter_rule(file_):
        file_=os.path.basename(file_).split('-')[-2]
        tokens = re.split("['+', '\-', \d]", file_)
        file_actions =  set(tokens) & __ALL_ACTIONS
        return file_actions <= allowed_actions

    files = glob.glob(f'{root_dir}/{kind}/*.ulx')
    return list(filter(filter_rule, files))


def get_nav_non_nav_files(root_dir, allowed_actions=__ALL_ACTIONS, kind='train'):
    train_files = get_files(root_dir, allowed_actions, kind=kind)
    print(f'Got {len(train_files)} files')
    navigation_condition = lambda x: 'go'in x
    navigation_train_files = [x for x in train_files if navigation_condition(os.path.basename(x).split('-')[-2])]
    non_navigation_train_files = [x for x in train_files if not navigation_condition(os.path.basename(x).split('-')[-2])]
    print(f'Got {len(train_files)} files')
    print(f'Got {len(navigation_train_files)} navigation files')
    print(f'Got {len(non_navigation_train_files)} non-navigation files')
    return train_files, navigation_train_files, non_navigation_train_files

def split_navigation_games(games):
    games_go6 = [x for x in games if "go6" in x]
    games_go9 = [x for x in games if "go9" in x]
    games_go12 = [x for x in games if "go12" in x]
    print('{} games split into go6: {}, go9: {}, go12: {}'.format(len(games), len(games_go6), len(games_go9), len(games_go12)))
    return games_go6, games_go9, games_go12
