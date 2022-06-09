import gym
import numpy as np
from gym import spaces
from collections import deque

import os
import cv2
import torch
import torch.nn.functional as F
import imageio

import textworld
import textworld.gym
from textworld import EnvInfos
from tw_cooking_game_puzzle.visual_hints import VisualHints
from tw_cooking_game_puzzle.utils import clean_str

from tqdm import tqdm
from utils import get_files, __ALL_ACTIONS, get_nav_non_nav_files

from matplotlib import pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
plt.rcParams['hatch.color'] = 'r'

import cv2
DEFAULT_HINT_LEVEL=3
class CookingNavigationEnv(gym.Env):
    def __init__(self, games, im_size=64, num_actions=5, stack_dim=None, time_penalty=False, 
                 name_type='literal', max_steps=30, use_multi_level_hints=False, hint_level=DEFAULT_HINT_LEVEL):
        print('Initializing environments')
        self.name_type=name_type
        self.buffer = deque(maxlen=4)

        self.im_size=im_size
        self.stack_dim=stack_dim
        self.time_penalty=time_penalty
        
        self.games=games
        self.num_games=len(games)
        self.counter=0
        self.max_steps=max_steps

        self.hint_level=hint_level
        self.use_multi_level_hints=use_multi_level_hints

        self.fig=plt.figure(figsize=(5,5), dpi= 300, facecolor='w', edgecolor='k')
        self.num_actions=num_actions
        
        if time_penalty:
            print('Using time penalty...')

        self.action_space = spaces.Discrete(num_actions)
        if stack_dim is not None:
            self.observation_space = spaces.Box(low=0, high=255, shape=(stack_dim, im_size, im_size, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(im_size, im_size, 3), dtype=np.uint8)
        
        self.request_infos = self.get_request_infos()

        print('**'*30)
        print("Using hint level: ", self.hint_level)
        print('**'*30)
        
    def get_request_infos(self):
        return EnvInfos(description=True, inventory=True, entities=True, verbs=True, facts=True, # feedback=True,
                        admissible_commands=True, intermediate_reward=True, score=True, max_score=True)
    
    def get_scores(self, infos):
        return infos['score']
    
        
    def reset(self, return_obs=False):
        self.ep_steps=0
        self.total_reward=0
        self.action = "reset"
        self.frame_buffer=[]
        self.game_name=self.games[self.counter]
        

        self.env = VisualHints(path=self.games[self.counter], request_infos=self.request_infos, batch_size=1, asynchronous=True, 
                               mask=False, distance_of_puzzle=4, add_death_room=False, clue_first_room=True,
                               max_number_inaccessible_rooms=2, room_name=True, color_way=True, upgradable_color_way=True,
                               name_type=self.name_type, draw_passages=True, draw_player=True, level_clue='easy', random_place=True, use_multi_level_hints=self.use_multi_level_hints, hint_level=self.hint_level,
                               name='cooking_game', max_episode_steps=self.max_steps)
        
        self.buffer = deque(maxlen=4)
        self.partial_pic=None
        obs_orig, rewards, dones, infos_orig, im, partial_pic, hint, indication_deathroom = self.env.reset()
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.step(['examine hint'])
        
        self.partial_pic=partial_pic
        partial_pic = F.interpolate(partial_pic, size=self.im_size)
        
        if self.stack_dim is not None:
            for i in range(4):
                self.buffer.append(partial_pic)
            state=torch.stack(self.buffer)
        else:
            state=partial_pic
        state = (np.transpose(state.numpy(), (0,2,3,1))[0]*255).astype('uint8')

        self.counter=(self.counter+1)%self.num_games
        if return_obs:
            return state, obs_orig, infos_orig
        else:
            return state
        
        
    def step(self, action, return_obs=False):
        self.ep_steps+=1
        if return_obs:
            self.action=action
        else:
            action_list = ["go north", "go south", "go east", "go west", "open door"]
            self.action=[action_list[action]]
        
        obs, rewards, dones, infos, im, partial_pic, _, _ = self.env.step(self.action)
        scores = infos['score']
        
        self.partial_pic=partial_pic
        partial_pic = F.interpolate(partial_pic, size=self.im_size)
        
        if ('-= Kitchen =-' in obs[0]) or ('-= Kitchen =-' in infos['description'][0]):
            if not return_obs:
                dones = [True] * len(obs)
                scores[0] += 1
        rew = scores[0]
        if self.time_penalty:
            rew-=0.02
            
        if self.stack_dim is not None:
            self.buffer.append(partial_pic)
            state=torch.stack(self.buffer)
        else:
            state=partial_pic
            
        state = (np.transpose(state.numpy(), (0,2,3,1))[0]*255).astype('uint8')
        self.total_reward += rew

        if return_obs:
            return state, obs, infos['score'], dones, infos
        else:
            return state, rew, dones[0], infos 

    def reset_textual(self):
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.reset()
        return obs, infos

    def step_textual(self, action):
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.step([action])
        scores = infos['score']
        return obs, scores, dones, infos
        
    
    def render(self, verbose=False, save_frame=False):
        partial_im = np.transpose(self.partial_pic.numpy(), (0,2,3,1))[0]
        partial_im = (partial_im*255).astype('uint8')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(partial_im, self.action, (10,10), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        
        if save_frame:
            self.frame_buffer.append(partial_im)
        return partial_im
    
    def end_episode(self):
        self.counter=(self.counter+1)%self.num_games

    def save_video(self, save_dir):
        filename = os.path.join(save_dir, os.path.basename(self.game_name).replace('.ulx', '.mp4'))
        print('Saving video to ', filename)
        writer = imageio.get_writer(filename, fps=2)
        for im in self.frame_buffer:
            writer.append_data(im)
        writer.close()


class CookingNavigationMultiHintsEnv(CookingNavigationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_multi_level_hints, "This option should be true to call this Env"
        assert self.num_actions==8, "Number of "

    def reset(self):
        self.ep_steps=0
        self.total_reward=0
        self.action = "reset"
        self.frame_buffer=[]
        self.game_name=self.games[self.counter]
        self.partial_pic=None
        self.full_partial_pic=None
        self.state=None

        self.env = VisualHints(path=self.games[self.counter], request_infos=self.request_infos, batch_size=1, asynchronous=True, 
                               mask=False, distance_of_puzzle=4, add_death_room=True, clue_first_room=True,
                               max_number_inaccessible_rooms=2, room_name=True, color_way=True, upgradable_color_way=True,
                               name_type=self.name_type, draw_passages=True, draw_player=True, level_clue='easy',random_place=True, use_multi_level_hints=self.use_multi_level_hints,
                               name='cooking_game', max_episode_steps=50)
        
        self.buffer = deque(maxlen=4)
        
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.reset()
        obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.step(['examine hint'])

        self.full_partial_pic=partial_pic
        self.partial_pic=self.full_partial_pic[...,0]

        self.partial_pic = F.interpolate(self.partial_pic, size=self.im_size)
        state = self.partial_pic
        state = (np.transpose(state.numpy(), (0,2,3,1))[0]*255).astype('uint8')
        self.state=state
        self.counter=(self.counter+1)%self.num_games

        return self.state
    
    def step(self, action):
        self.ep_steps+=1
        action_list = ["go north", "go south", "go east", "go west", "open door"]
        if action<=4:
            self.action=action_list[action]
            obs, rewards, dones, infos, im, partial_pic, hint, indication_deathroom = self.env.step([action_list[action]])
            scores = infos['score']
            self.full_partial_pic=partial_pic
            if ('-= Kitchen =-' in obs[0]) or ('-= Kitchen =-' in infos['description'][0]):
                dones = [True] * len(obs)
                scores[0] += 1
            rew = scores[0]
        else:
            hint_num = action-5
            self.partial_pic=self.full_partial_pic[...,hint_num]
            self.partial_pic = F.interpolate(self.partial_pic, size=self.im_size)
            state = self.partial_pic
            state = (np.transpose(state.numpy(), (0,2,3,1))[0]*255).astype('uint8')
            self.state=state
            dones = [False]
            infos = {}
            rew={0:-0.01, 1:-0.02, 2:-0.05}[hint_num]
        self.total_reward += rew
        return self.state, rew, dones[0], infos 
        
    def render(self, verbose=False, save_frame=False):
        partial_im = np.transpose(self.full_partial_pic[...,2].numpy(), (0,2,3,1))[0]
        partial_im = (partial_im*255).astype('uint8')
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(partial_im, self.action, (10,10), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        if save_frame:
            self.frame_buffer.append(partial_im)
        return partial_im


if __name__=='__main__':
    root_dir="/home/subhajit/Data/text_world_compete/"
    allowed_actions = __ALL_ACTIONS
    train_files, navigation_train_files, non_navigation_train_files = get_nav_non_nav_files(root_dir=root_dir,allowed_actions=allowed_actions)
    env = CookingNavigationEnv(navigation_train_files, im_size=64, num_actions=5, stack_dim=None, 
                               time_penalty=True, name_type='literal')
 
