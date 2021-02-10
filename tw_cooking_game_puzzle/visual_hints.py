import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")
import gym
import textworld
import textworld.gym

import torch
import torchvision.transforms as transforms

from PIL import ImageOps
transform = transforms.Compose([transforms.ToTensor()])
from tw_cooking_game_puzzle.cooking_map_builder import build_dict_rooms, build_dict_game_goals, picture_size, draw_map, pic_player_to_kitchen
from textworld import EnvInfos

class VisualHintsWrapper:

    def __init__(self, env, batch_size: int,
                 mask: bool, distance_of_puzzle: int, clue_first_room: bool, add_death_room: bool,
                 max_number_inaccessible_rooms: int, room_name=True, color_way=True, upgradable_color_way=True,
                 name_type=["literal", 'random_numbers', 'room_importance'], draw_passages=True, draw_player=True,
                 level_clue=['easy', 'medium', 'hard', 'very hard'], random_place=False, name="map"):
        """
        Create Role Master
        The Role Master is an intermediate between the environment and the agent
        He can add extra information and display some puzzles

        Agent <-> Role Master <-> environment
        """

        self.env = env
        #self.infos = []
        self.infos = {}

        self.batch_size = batch_size
        self.rooms_dict = dict()
        self.mask = mask  # If you want to mask all  rooms excepted the ones in the colorway
        self.distance_of_puzzle = distance_of_puzzle  # if clue_first_room room distance of the puzzle is no considered
        self.clue_first_room = clue_first_room
        self.add_death_room = add_death_room
        self.death_room = None
        self.rooms_leading_to_death_room = dict()
        self.beginning_room = None
        self.max_number_inaccessible_rooms = max_number_inaccessible_rooms
        self.room_name = room_name
        self.color_way = color_way
        self.upgradable_color_way = upgradable_color_way
        self.name_type = name_type
        self.draw_passages = draw_passages
        self.draw_player = draw_player
        self.level_clue = level_clue
        self.random_place = random_place
        self.name = name

        self.im = torch.zeros(3, 500, 500, dtype=torch.float)  # the picture in the hint
        self.no_im = torch.zeros(3, 500, 500, dtype=torch.float)  # the picture in the hint

        self.hint = ''  # the hint to understand the picture by the agent
        self.indication_deathroom = ''  # the clue to avoid the death_room by the agent
        self.take_hint = [False] * batch_size  # the memory that the agent has found and has taken the hint
        self.read_hint = [False] * batch_size  # the memory of the hint to understand the picture by the agent
        self.read_indication_deathroom = [False] * batch_size  # the memory of the clue to avoid the death_room by the
        # agent

        self.current_room = ['']*batch_size
        self.room_of_the_hint = -1
        self.level_clue = level_clue

        self.rewards_hint = [1] * self.batch_size
        self.rewards_board = [1] * self.batch_size
        self.rewards_death = [-1] * self.batch_size

        self.vect_infos_obs_change = np.vectorize(self.infos_obs_change, excluded=['room_of_the_hint', 'beginning'])
        self.vect_need_step_mod = np.vectorize(self.need_step_mod, excluded=['need_step_mod'])
        self.vect_step_mod = np.vectorize(self.step_mod)

    def reset(self):
        '''
        reset the environment
        :return: observations, informations of env.reset() possibly modify by infos_obs_change
        '''
        obs, infos = self.env.reset()

        # build the picture and the clue
        self.rooms_dict, first_room = build_dict_rooms(infos)
        self.dict_game_goals = build_dict_game_goals(infos)
        self.beginning_room = self.dict_game_goals['at']['P']
        self.current_room = [self.beginning_room]*self.batch_size

        # size of the picture
        visited_rooms = dict()
        self.center_visited_rooms = dict()

        for k in iter(self.rooms_dict):
            visited_rooms[k] = False
            self.center_visited_rooms[k] = np.zeros(2)

        self.pic_size, self.center_visited_rooms = picture_size(self.rooms_dict, first_room, -1, visited_rooms,
                                                                self.center_visited_rooms)

        # draw the map
        way, self.death_room, self.rooms_leading_to_death_room, self.dict_rooms_nbr, _, self.im = draw_map(
            self.pic_size,
            self.center_visited_rooms,
            self.rooms_dict,
            self.dict_game_goals,
            mask=self.mask,
            distance_of_puzzle=self.distance_of_puzzle,
            clue_first_room=self.clue_first_room,
            add_death_room=self.add_death_room,
            max_number_inaccessible_rooms=self.max_number_inaccessible_rooms,
            room_name=self.room_name,
            color_way=self.room_name,
            name_type=self.name_type,
            draw_passages=self.draw_passages,
            draw_player=self.draw_player,
            random_place=self.random_place,
            name=self.name)

        # pad image and convert to tensor
        im_size = self.im.size
        desired_size = 500

        delta_w = desired_size - im_size[0]
        delta_h = desired_size - im_size[1]

        padding = (delta_w, delta_h, 0, 0)
        new_im = ImageOps.expand(self.im, padding)

        self.im = transform(new_im).float()

        # load the hint
        role_master_file = open(self.name + '_role_master.txt', 'r+')
        f_line = role_master_file.readlines()

        b_clue = False
        len_f_line = len(f_line)
        cl = ''

        for i in range(len_f_line):
            if b_clue and self.level_clue in f_line[i]:
                cl = f_line[i + 1]
                break
            if 'clue' in f_line[i]:
                b_clue = True
        self.hint = cl
        role_master_file.close()
        # load the indication to avoid the death room
        role_master_file = open(self.name + '_role_master.txt', 'r+')
        f_line = role_master_file.readlines()

        b_clue = False
        len_f_line = len(f_line)
        cl = ''

        for i in range(len_f_line):

            if b_clue and self.level_clue in f_line[i]:
                cl = f_line[i + 1]
                break
            if 'death room' in f_line[i]:
                b_clue = True

        self.indication_deathroom = cl

        role_master_file.close()

        self.room_of_the_hint = way[-1]

        obs, infos_description = self.vect_infos_obs_change(obs, infos["description"],
                                                            room_of_the_hint=self.room_of_the_hint,
                                                            beginning=self.beginning_room,
                                                            take_hint=self.take_hint)

        self.infos = infos
        self.infos['description'] = infos_description
        im = [self.no_im] * self.batch_size
        im = torch.stack(im)

        return obs, [0] * self.batch_size, [False] * self.batch_size, self.infos, im, torch.stack(
            [self.no_im] * self.batch_size), [''] * self.batch_size, [''] * self.batch_size

    def infos_obs_change(self, obs: tuple, infos_description: list, room_of_the_hint: int, beginning: str,
                         take_hint: bool):
        '''

        :param beginning: the room where the player begins
        :param obs: observations from the environment
        :param infos_description: description of the environment
        :param room_of_the_hint: name of the room where is placed the hint
        :return: modify obs and infos depending on the place of the player
        '''

        if (beginning in obs or beginning in infos_description.lower()) and self.death_room is not None:
            if '.\n\nThere is' in obs:
                idx = obs.index('.\n\nThere is')
                obs = obs[:idx] + '. There is a board on the wall' + obs[idx:]
                idx = infos_description.index('.\n\nThere is')
                infos_description = infos_description[:idx] + '. There is a board on the wall' + \
                                    infos_description[idx:]
            else:
                obs = obs + '. There is a board on the wall'
                infos_description = infos_description + '. There is a board on the wall'

        if (room_of_the_hint in obs or room_of_the_hint in infos_description.lower()) and take_hint is False:
            if '.\n\nThere is' in obs:
                idx = obs.index('.\n\nThere is')
                obs = obs[:idx] + '. There is a hint on the floor' + obs[idx:]
                idx = infos_description.index('.\n\nThere is')
                infos_description = infos_description[:idx] + '. There is a hint on the floor' + \
                                    infos_description[idx:]
            else:
                obs = obs + '. There is a hint on the floor'
                infos_description = infos_description + '. There is a hint on the floor'

        return obs, infos_description

    def need_step_mod(self, infos_description, take_hint, step):

        # agent is in a room next to death room  orders to go in death room
        lethal_step = False
        for k in iter(self.rooms_leading_to_death_room):
            if self.rooms_leading_to_death_room[k] in infos_description and 'go ' + k in step:
                lethal_step = True

        if self.room_of_the_hint in infos_description and take_hint is False and ('take hint' == step
                                                                                  or ' take the hint' == step):
            return 1

        if take_hint and ('read the hint' == step
                          or 'read hint' == step
                          or 'look the hint' == step
                          or 'look at the hint' == step
                          or 'look hint' == step):
            return 2

        elif 'board' in infos_description and ('read the board' == step
                                               or 'read board' == step
                                               or 'look the board' == step
                                               or 'look at the board' == step
                                               or 'look board' == step):

            return 2

        elif self.room_of_the_hint in infos_description and take_hint is False and ('examine hint' == step
                                                                                    or ' examine the hint' == step):
            return 3

        elif self.death_room is not None and lethal_step:
            return 4  # dead state

        elif self.death_room is not None and 'The End' in infos_description:
            return 4  # dead state

        else:
            return 0

    def step(self, step):
        # self.infos should has 'description' as set by reset()
        info_des = self.infos['description'] if 'description' in self.infos else ''

        need_step_mod = self.vect_need_step_mod(info_des, self.take_hint, step)
        obs = np.asarray([])
        rewards = np.asarray([])
        dones = np.asarray([])
        infos = dict()
        for k in iter(self.infos):
            infos[k] = []
        im = None
        hint = None
        indication_deathroom = None
        i = 0
        while i < self.batch_size:
            j = i
            b = need_step_mod[j]
            sample = [step[i]]
            while j + 1 < self.batch_size and b == need_step_mod[j + 1]:
                sample.append(step[j + 1])
                j += 1

            if b == 1:
                info_des = self.infos['description'][i:j + 1]
                rewards_hint = self.rewards_hint[i:j + 1]
                rewards_board = self.rewards_board[i:j + 1]
                obs_env, rewards_env, rewards_hint, rewards_board, dones_env, infos_feedback, take_hint_env, im_env, \
                hint_env, indic_dr_env = self.vect_step_mod(b, info_des, sample, rewards_hint, rewards_board)

                self.rewards_hint[i:j + 1] = rewards_hint
                self.take_hint[i:j + 1] = take_hint_env

                len_sample = len(sample)
                for k in iter(self.infos):
                    for l in range(len_sample):
                        if k == 'feedback':
                            infos[k].append(infos_feedback[l])
                        elif k == 'last_command':
                            infos[k].append(step[i + l])
                        elif k == 'moves':
                            infos[k].append(self.infos[k][i + l] + 1)
                        elif k == 'intermediate_reward':
                            infos[k].append(rewards_env[l])
                        elif k == 'score':
                            infos[k].append(self.infos[k][i + l])
                        else:
                            infos[k].append(self.infos[k][i + l])

                obs = np.concatenate((obs, obs_env), axis=0)
                rewards = np.concatenate((rewards, rewards_env), axis=0)
                dones = np.concatenate((dones, dones_env), axis=0)

                for j_env in range(0, j + 1 - i):

                    if im_env[j_env] or self.read_hint[i + j_env]:
                        self.read_hint[j_env] = True
                        if im is None:
                            im = [self.im]
                        else:
                            im = im + [self.im]

                        if hint is None:
                            hint = [self.hint]
                        else:
                            hint = hint + [self.hint]

                    elif im_env[j_env] == False and self.read_hint[i + j_env] == False:
                        if im is None:
                            im = [self.no_im]
                        else:
                            im = im + [self.no_im]

                        if hint is None:
                            hint = ['']
                        else:
                            hint = hint + ['']

                for k_env in range(0, j + 1 - i):

                    if indic_dr_env[k_env] or self.read_indication_deathroom[i + k_env]:

                        self.read_indication_deathroom[i + k_env] = True
                        if indication_deathroom is None:
                            indication_deathroom = [self.indication_deathroom]
                        else:
                            indication_deathroom = indication_deathroom + [self.indication_deathroom]

                    elif indic_dr_env[k_env] == False and self.read_indication_deathroom[i + k_env] == False:
                        if indication_deathroom is None:
                            indication_deathroom = ['']
                        else:
                            indication_deathroom = indication_deathroom + ['']

            if b == 2:
                info_des = self.infos['description'][i:j + 1]
                rewards_hint = self.rewards_hint[i:j + 1]
                rewards_board = self.rewards_board[i:j + 1]
                obs_env, rewards_env, rewards_hint, rewards_board, dones_env, infos_feedback, take_hint_env, im_env, \
                hint_env, indic_dr_env = self.vect_step_mod(b, info_des, sample, rewards_hint, rewards_board)

                self.rewards_hint[i:j + 1] = rewards_hint
                self.rewards_board[i:j + 1] = rewards_board

                len_sample = len(sample)
                for k in iter(self.infos):
                    for l in range(len_sample):
                        if k == 'feedback':
                            infos[k].append(infos_feedback[l])
                        elif k == 'last_command':
                            infos[k].append(step[i + l])
                        elif k == 'moves':
                            infos[k].append(self.infos[k][i + l] + 1)
                        elif k == 'intermediate_reward':
                            infos[k].append(rewards_env[l])
                        elif k == 'score':
                            infos[k].append(self.infos[k][i + l])
                        else:
                            infos[k].append(self.infos[k][i + l])

                obs = np.concatenate((obs, obs_env), axis=0)
                rewards = np.concatenate((rewards, rewards_env), axis=0)
                dones = np.concatenate((dones, dones_env), axis=0)

                for j_env in range(0, j + 1 - i):

                    if im_env[j_env] or self.read_hint[i + j_env]:
                        self.read_hint[j_env] = True
                        if im is None:
                            im = [self.im]
                        else:
                            im = im + [self.im]

                        if hint is None:
                            hint = [self.hint]
                        else:
                            hint = hint + [self.hint]

                    elif im_env[j_env] == False and self.read_hint[i + j_env] == False:
                        if im is None:
                            im = [self.no_im]
                        else:
                            im = im + [self.no_im]

                        if hint is None:
                            hint = ['']
                        else:
                            hint = hint + ['']

                for k_env in range(0, j + 1 - i):

                    if indic_dr_env[k_env] or self.read_indication_deathroom[i + k_env]:

                        self.read_indication_deathroom[i + k_env] = True
                        if indication_deathroom is None:
                            indication_deathroom = [self.indication_deathroom]
                        else:
                            indication_deathroom = indication_deathroom + [self.indication_deathroom]

                    elif indic_dr_env[k_env] == False and self.read_indication_deathroom[i + k_env] == False:
                        if indication_deathroom is None:
                            indication_deathroom = ['']
                        else:
                            indication_deathroom = indication_deathroom + ['']

            if b == 3:
                info_des = self.infos['description'][i:j + 1]
                rewards_hint = self.rewards_hint[i:j + 1]
                rewards_board = self.rewards_board[i:j + 1]
                obs_env, rewards_env, rewards_hint, rewards_board, dones_env, infos_feedback, take_hint_env, im_env, \
                hint_env, indic_dr_env = self.vect_step_mod(b, info_des, sample, rewards_hint, rewards_board)

                self.rewards_hint[i:j + 1] = rewards_hint
                self.take_hint[i:j + 1] = take_hint_env

                len_sample = len(sample)
                for k in iter(self.infos):
                    for l in range(len_sample):
                        if k == 'feedback':
                            infos[k].append(infos_feedback[l])
                        elif k == 'last_command':
                            infos[k].append(step[i + l])
                        elif k == 'moves':
                            infos[k].append(self.infos[k][i + l] + 1)
                        elif k == 'intermediate_reward':
                            infos[k].append(rewards_env[l])
                        elif k == 'score':
                            infos[k].append(self.infos[k][i + l])
                        else:
                            infos[k].append(self.infos[k][i + l])

                obs = np.concatenate((obs, obs_env), axis=0)
                rewards = np.concatenate((rewards, rewards_env), axis=0)
                dones = np.concatenate((dones, dones_env), axis=0)

                for j_env in range(0, j + 1 - i):

                    if im_env[j_env] or self.read_hint[i + j_env]:
                        self.read_hint[j_env] = True
                        if im is None:
                            im = [self.im]
                        else:
                            im = im + [self.im]

                        if hint is None:
                            hint = [self.hint]
                        else:
                            hint = hint + [self.hint]

                    elif im_env[j_env] == False and self.read_hint[i + j_env] == False:
                        if im is None:
                            im = [self.no_im]
                        else:
                            im = im + [self.no_im]

                        if hint is None:
                            hint = ['']
                        else:
                            hint = hint + ['']

                for k_env in range(0, j + 1 - i):

                    if indic_dr_env[k_env] or self.read_indication_deathroom[i + k_env]:

                        self.read_indication_deathroom[i + k_env] = True
                        if indication_deathroom is None:
                            indication_deathroom = [self.indication_deathroom]
                        else:
                            indication_deathroom = indication_deathroom + [self.indication_deathroom]

                    elif indic_dr_env[k_env] == False and self.read_indication_deathroom[i + k_env] == False:
                        if indication_deathroom is None:
                            indication_deathroom = ['']
                        else:
                            indication_deathroom = indication_deathroom + ['']

            elif b == 0:

                completed_sample = [' '] * i + sample + [' '] * (self.batch_size - (j + 1))
                obs_env, rewards_env, dones_env, infos_env = self.env.step(completed_sample)

                obs_env = obs_env[i:j + 1]
                rewards_env = rewards_env[i:j + 1]
                dones_env = dones_env[i:j + 1]

                obs_env, infos_description = self.vect_infos_obs_change(obs_env, infos_env["description"][i:j + 1],
                                                                        room_of_the_hint=self.room_of_the_hint,
                                                                        beginning=self.beginning_room,
                                                                        take_hint=self.take_hint[i:j + 1])

                infos_env['description'][i:j + 1] = infos_description

                obs = np.concatenate((obs, obs_env), axis=0)
                rewards = np.concatenate((rewards, rewards_env), axis=0)
                dones = np.concatenate((dones, dones_env), axis=0)
                len_sample = len(sample)

                for k in iter(self.infos):
                    for l in range(len_sample):
                        if k == 'intermediate_reward':
                            infos[k].append(0)
                        elif k == 'score':
                            infos[k].append(rewards_env[l])
                        else:
                            infos[k].append(infos_env[k][l])

                for j_env in range(i, j + 1):
                    if self.read_hint[j_env]:

                        if im is None:
                            im = [self.im]
                        else:
                            im = im + [self.im]

                        if hint is None:
                            hint = [self.hint]
                        else:
                            hint = hint + [self.hint]

                    elif self.read_hint[j_env] == False:

                        if im is None:
                            im = [self.no_im]
                        else:
                            im = im + [self.no_im]

                        if hint is None:
                            hint = ['']
                        else:
                            hint = hint + ['']
                for k_env in range(i, j + 1):

                    if self.read_indication_deathroom[k_env]:

                        if indication_deathroom is None:
                            indication_deathroom = [self.indication_deathroom]
                        else:
                            indication_deathroom = indication_deathroom + [self.indication_deathroom]

                    elif self.read_indication_deathroom[k_env] == False:

                        if indication_deathroom is None:
                            indication_deathroom = ['']
                        else:
                            indication_deathroom = indication_deathroom + ['']


            elif b == 4:
                len_sample = len(sample)
                for l in range(len_sample):
                    obs = np.concatenate((obs, ['you die! :\r\n']), axis=0)
                    r = self.rewards_death[i + l]
                    self.rewards_death[i + l] = np.minimum(0, self.rewards_death[i + l] + 1)
                    rewards = np.concatenate((rewards, [r]), axis=0)
                    dones = np.concatenate((dones, [True]), axis=0)

                for j_env in range(i, j + 1):
                    if self.read_hint[j_env]:

                        if im is None:
                            im = [self.im]
                        else:
                            im = im + [self.im]

                        if hint is None:
                            hint = [self.hint]
                        else:
                            hint = hint + [self.hint]

                    elif self.read_hint[j_env] == False:

                        if im is None:
                            im = [self.no_im]
                        else:
                            im = im + [self.no_im]

                        if hint is None:
                            hint = ['']
                        else:
                            hint = hint + ['']

                for k_env in range(i, j + 1):

                    if self.read_indication_deathroom[k_env]:

                        if indication_deathroom is None:

                            indication_deathroom = [self.indication_deathroom]

                        else:

                            indication_deathroom = indication_deathroom + [self.indication_deathroom]


                    elif self.read_indication_deathroom[k_env] == False:

                        if indication_deathroom is None:

                            indication_deathroom = ['']

                        else:

                            indication_deathroom = indication_deathroom + ['']

                for k in iter(self.infos):
                    for l in range(len_sample):
                        if k == 'intermediate_reward':

                            infos[k].append(int(rewards[i + l]))
                        elif k == 'description':
                            infos[k].append('-= ' + self.death_room + ' =-*** The End ***')
                        elif k == 'feedback':
                            infos[k].append('you die! :\r\n')
                        elif k == 'last_command':
                            infos[k].append(None)
                        elif k == 'moves':
                            infos[k].append(self.infos[k][i + l] + 1)
                        elif k == 'score':
                            infos[k].append(self.infos[k][i + l])
                        else:
                            infos[k].append(self.infos[k][i + l])

            i = j + 1

        info_des = infos['description']

        if self.upgradable_color_way:

            partial_view_batch = []
            for i_batch in range(self.batch_size):

                self.current_room[i_batch] = info_des[i_batch].split('-= ')[1].split(' =-')[0].lower()

                if self.rewards_hint[i_batch] <= 0:
                    partial_view_pic = pic_player_to_kitchen(self.rooms_dict, self.dict_game_goals,
                                                             self.center_visited_rooms, self.pic_size,
                                                             self.current_room[i_batch], room_name=self.room_name,
                                                             dict_rooms_numbers=self.dict_rooms_nbr,
                                                             name_type=self.name_type,
                                                             draw_passages=self.draw_passages,
                                                             draw_player=self.draw_player)

                    partial_view_pic_size = partial_view_pic.size
                    desired_size = 500

                    delta_w = desired_size - partial_view_pic_size[0]
                    delta_h = desired_size - partial_view_pic_size[1]

                    padding = (delta_w, delta_h, 0, 0)
                    new_im = ImageOps.expand(partial_view_pic, padding)

                    partial_view_batch.append(transform(new_im).float())

                else:
                    partial_view_batch.append(self.no_im)

            partial_view_batch = torch.stack(partial_view_batch)

            self.infos = infos
            im = torch.stack(im)

            return obs, rewards, dones, infos, im, partial_view_batch, hint, indication_deathroom

        self.infos = infos
        im = torch.stack(im)

        return obs, rewards, dones, infos, im, torch.stack([self.no_im] * self.batch_size), hint, indication_deathroom

    def step_mod(self, need_step_mod, infos_description, step, rewards_hint, rewards_board):
        """
        :param step: the step that the agents want to do
        :return: special observations, rewards for having read the
        hint, reward for having read the board, done, infos_description, if there is an image displayed (Bool),
        a hint read(Bool), an indication to avoid the death room(Bool)
        """

        room = infos_description.split('-= ')[1].split(' =-')[0]
        if need_step_mod == 1:

            obs = 'You have picked up thye hint. Reading it reveals the visual hint!\r\n'
            infos_description = ' -= ' + room + ' =- ' + obs
            dones = False
            return obs, 0, rewards_hint, rewards_board, dones, infos_description, True, False, False, False

        elif need_step_mod == 2 and (
                'read the hint' == step or 'read hint' == step
                or 'look the hint' == step or 'look at the hint' == step or 'look hint' == step):

            obs = 'You have accessed the visual hint! Follow the textual advice:\r\n' + self.hint
            infos_description = ' -= ' + room + ' =- ' + obs
            rewards = rewards_hint
            rewards_hint = np.maximum(0, rewards_hint - 1)  # rewarded only for the first reading
            dones = False
            return obs, rewards, rewards_hint, rewards_board, dones, infos_description, False, True, True, False

        elif need_step_mod == 3:

            obs = 'You have accessed the visual hint! Follow the textual advice:\r\n' + self.hint
            infos_description = ' -= ' + room + ' =- ' + obs
            rewards = rewards_hint
            rewards_hint = np.maximum(0, rewards_hint - 1)  # rewarded only for the first reading
            dones = False
            return obs, rewards, rewards_hint, rewards_board, dones, infos_description, True, True, True, False

        elif 'board' in infos_description and ('read the board' == step
                                               or 'read board' == step
                                               or 'look the board' == step
                                               or 'look at the board' == step
                                               or 'look board' == step):

            obs = 'Be careful! This information can save you:\r\n' + self.indication_deathroom
            infos_des = ' -= ' + room + ' =- ' + obs
            rewards = rewards_board
            rewards_board = np.maximum(0, rewards_board - 1)  # rewarded only for the first reading
            dones = False

            return obs, rewards, rewards_hint, rewards_board, dones, infos_description, False, False, False, True


        elif self.death_room is not None and self.death_room in infos_description:

            obs = ('You die! :\r\n',)
            infos_description = ' -= ' + room + ' =- ' + obs
            rewards = -1
            dones = True

            return obs, rewards, rewards_hint, rewards_board, dones, infos_description, False, False, False, False


class VisualHints(VisualHintsWrapper):

    def __init__(self, path: str, request_infos: EnvInfos, batch_size: int, asynchronous: bool,
                 *args, **kwargs):
        """
        Create Role Master
        The Role Master is an intermediate between the environment and the agent
        He can add extra information and display some puzzles

        Agent <-> Role Master <-> environment
        """
        n_id = textworld.gym.register_game(path,
                                   batch_size=batch_size,
                                   asynchronous=asynchronous,
                                   request_infos=request_infos,
                                   )
        env = gym.make(n_id)
        env.seed(42)
        env.display_command_during_render = True
        super(VisualHints, self).__init__(env, *args, batch_size=batch_size, **kwargs)
