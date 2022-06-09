import os
import numpy as np
import argparse
from os.path import join as pjoin
import sys

sys.path.append(sys.path[0] + "/..")

import gym
import textworld
import textworld.gym

from tw_cooking_game_puzzle.visual_hints import VisualHints
from tw_cooking_game_puzzle.utils import clean_str

from textworld import EnvInfos
from pprint import pprint

from collections import deque
from tqdm import tqdm
import pickle

import torchvision
os.makedirs('figs', exist_ok=True)

from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
plt.rcParams['hatch.color'] = 'r'

import glob
datadir='env_files'
env_files = glob.glob(datadir+'/*.ulx')
print(env_files)
import cv2
os.makedirs('figs/paper_figs', exist_ok=True)

from kitchen_env import CookingNavigationEnv, CookingNavigationMultiHintsEnv
from utils import get_files, __ALL_ACTIONS, get_nav_non_nav_files, split_navigation_games


root_dir = os.path.expanduser("~/Data/text_world_compete/ftwp/games/")

print('##'*30)
allowed_actions = __ALL_ACTIONS #- {"open"}

test_files, test_navigation_files, eval_non_navigation_files = get_nav_non_nav_files(root_dir=root_dir,allowed_actions=allowed_actions, kind='test')

test_navigation_games_go6, test_navigation_games_go9, test_navigation_games_go12 = split_navigation_games(test_navigation_files)
files_dict={'go6': test_navigation_games_go6, 'go9': test_navigation_games_go9,  'go12': test_navigation_games_go12}

for game_type, game_list in files_dict.items():
    os.makedirs('figs/paper_figs_{}'.format(game_type), exist_ok=True)
    for hint_level in [1,2,3,4]:
        env = CookingNavigationEnv(game_list, im_size=500, num_actions=5, stack_dim=None, time_penalty=False, hint_level=hint_level)
        env_multi = CookingNavigationMultiHintsEnv(game_list, im_size=500, num_actions=8, stack_dim=None, time_penalty=False, use_multi_level_hints=True)
        print('*** Hint level {} ***'.format(hint_level))
        for k in range(25):
            print('*** Game num {} ***'.format(k))
            if hint_level<=3:
                st, obs, infos = env.reset(return_obs=True)
            else:
                st = env_multi.reset()
                st, _, _, _ =env_multi.step(7)
            cv2.imwrite('figs/paper_figs_{}/im_{}_hint_level_{}.png'.format(game_type, k+1, hint_level), st[:,:,::-1].astype('uint8'))

