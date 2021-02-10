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

from textworld.render.graph import show_graph
from pprint import pprint


def get_next(env, action=None, postfix='', save_im=False, partial=False, image_savedir='figs'):
    # Get the next command: if action is None then reset the environment
    plt.clf()
    if action==None:
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = env.reset()
    else:
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = env.step([action])
    if save_im:
        if partial:
            torchvision.utils.save_image(partial_pic, "{}/saved_image_{}.png".format(image_savedir, postfix), normalize=True)
        else:
            torchvision.utils.save_image(im, "{}/saved_image_{}.png".format(image_savedir, postfix), normalize=True)
    im = np.transpose(im.numpy(), (0,2,3,1))[0]
    partial_im = np.transpose(partial_pic.numpy(), (0,2,3,1))[0]
    h, w, c = im.shape
    return infos['description'][0], obs[0], hint[0], im, partial_im


def test_VisualHints():
    request_infos = EnvInfos(verbs=True, moves=True, inventory=True, description=True, objective=True, feedback=True,
                             intermediate_reward=True, facts=True, policy_commands=True)
    path = './tests/tw_games/tw-cooking-test-take+cook+cut+open+go6-JrmLfNyMcErjF6LD.ulx'
    env = VisualHints(path=path, request_infos=request_infos, batch_size=1, asynchronous=True,
                    mask=False, distance_of_puzzle=4, add_death_room=False, clue_first_room=True,
                    max_number_inaccessible_rooms=2, room_name=True, color_way=True, upgradable_color_way=True,
                    name_type='literal', draw_passages=True, draw_player=True, level_clue='easy', random_place=True,
                    name='cooking_game')

    obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = env.reset()
    actions = ["examine hint", "look", "open glass door", "go west", "go south", "open door", "go south", "go west", "examine cookbook", "inventory"]
    for action in actions:
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = env.step([action])
