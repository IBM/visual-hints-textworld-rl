
def write_clue(name_file, way, rooms_dict: dict, dict_game_goals: dict, dict_rooms_numbers: dict, death_room: str,
               name_type=["literal", 'random_numbers', 'room_importance']):
    '''
    write a txt file for that will be used by the game master class

    :param death_room: if there is a death_room
    :param name_file: the name of the file
    :param way: a table that gives the way between the cooking place and the room where is the hint
    :param rooms_dict:dictionary of the rooms see 'def build_dict_rooms' in cooking_map_builder
    :param dict_game_goals: a dictionary with the main goals and place of important elements for the game
                            see( def build_dict_game_goals) in cooking_map_builder
    :param dict_rooms_numbers: a dictionary with the number of each room in the mode 'random_numbers'
                               see( def draw_map) in cooking_map_builder
    :param name_type: three possibilities 'literal' : the true name of the room (eg. kitchen, bedroom)
                                          'random_numbers': a random number is attributed to each room
                                          'room_importance': each room receive a number based on the importance of the room
                     see( def draw_map) in cooking_map_builder
    :return:
    '''
    list_rooms_s_goals = []
    if 'secondary_goals' in dict_game_goals:
        for sgoal in iter(dict_game_goals['secondary_goals']):
            if dict_game_goals['secondary_goals'][sgoal] not in list_rooms_s_goals:
                list_rooms_s_goals.append(dict_game_goals['secondary_goals'][sgoal])

    f = open(name_file, "w+")
    f.write('####clue####\r\n')
    if name_type == 'literal':
        f.write('\r \n')
        f.write('easy:\r')

        clue = 'you are in the {}, take the ingredients in '.format(way[-1])
        if len(list_rooms_s_goals) == 0:
            clue += 'inventory,'
        else:
            list_rooms =[]
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the {}, '.format(dict_game_goals['secondary_goals'][sgoal])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the {}, '.format(dict_game_goals['cooking_location'])
        if death_room is not None:
            clue = clue + 'and avoid the death room which is the {}'.format(death_room)
        clue = clue+'\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('medium:\r')

        clue = 'take the ingredients in '
        if len(list_rooms_s_goals) == 0:
            clue += 'inventory,'
        else:
            list_rooms = []
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the {}, '.format(dict_game_goals['secondary_goals'][sgoal])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the {}, '.format(dict_game_goals['cooking_location'])
        if death_room is not None:
            clue = clue + 'and avoid the death room which is the {}'.format(death_room)
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('hard:\r')

        clue = 'take the ingredients in '
        if len(list_rooms_s_goals) == 0:
            clue += 'inventory,'
        else:
            list_rooms = []
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the {}, '.format(dict_game_goals['secondary_goals'][sgoal])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the {}'.format(dict_game_goals['cooking_location'])

        if len(rooms_dict) != 1:
            clue = clue +', note that '
            uninteresting_place = False
            for k in rooms_dict:
                if k not in list_rooms_s_goals and k != dict_game_goals['cooking_location']:
                    clue = clue + 'the {}, '.format(k)
                    uninteresting_place = True
            if uninteresting_place:
                clue = clue[:-2] + ' are uninteresting places'
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('very hard:\r')
        clue = ''

        if len(rooms_dict) != 1:
            uninteresting_place = False
            for k in rooms_dict:
                if k not in list_rooms_s_goals and k != dict_game_goals['cooking_location']:
                    clue = clue + 'the {}, '.format(k)
                    uninteresting_place = True
            if uninteresting_place:
                clue = clue[:-2] + ' are uninteresting places'
        else:
            clue = clue + 'there is only one room'
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\n')

    if name_type == 'random_numbers':

        f.write('\n')
        f.write('easy:\r')

        clue = 'you are in the room {}, take the ingredients in '.format(dict_rooms_numbers[way[-1]])
        if len(list_rooms_s_goals) == 0:
            clue += 'I,'
        else:
            list_rooms = []
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the room {}, '.format(dict_rooms_numbers[dict_game_goals['secondary_goals'][sgoal]])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the room {}, '.format(dict_rooms_numbers[dict_game_goals['cooking_location']])
        if death_room is not None:
            clue = clue + 'and avoid the death room which is the {}'.format(dict_rooms_numbers[death_room])
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('medium:\r')

        clue = 'take the ingredients in '
        if len(list_rooms_s_goals) == 0:
            clue += 'inventory,'
        else:
            list_rooms = []
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the room {}, '.format(dict_rooms_numbers[dict_game_goals['secondary_goals'][sgoal]])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the room {}, '.format(dict_rooms_numbers[dict_game_goals['cooking_location']])
        if death_room is not None:
            clue = clue + 'and avoid the death room which is the {}'.format(dict_rooms_numbers[death_room])
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('hard:\r')

        clue = 'take the ingredients in '
        if len(list_rooms_s_goals) == 0:
            clue += 'inventory,'
        else:
            list_rooms = []
            for sgoal in iter(dict_game_goals['secondary_goals']):
                if dict_game_goals['secondary_goals'][sgoal] not in list_rooms:
                    clue = clue + 'the room {}, '.format(dict_rooms_numbers[dict_game_goals['secondary_goals'][sgoal]])
                    list_rooms.append(dict_game_goals['secondary_goals'][sgoal])
        clue = clue + 'and cook in the room {}'.format(dict_rooms_numbers[dict_game_goals['cooking_location']])

        if len(rooms_dict) != 1:
            clue = clue + ', note that '
            uninteresting_place = False
            for k in rooms_dict:
                if k not in list_rooms_s_goals and k != dict_game_goals['cooking_location']:
                    clue = clue + 'the room {}, '.format(dict_rooms_numbers[k])
                    uninteresting_place = True
            if uninteresting_place:
                clue = clue[:-2] + ' are uninteresting places'
        clue = clue + '\r\n'
        f.write(clue)

        f.write('\r \n')
        f.write('very hard:\r')
        clue = ''

        if len(rooms_dict) != 1:
            uninteresting_place = False
            for k in rooms_dict:
                if k not in list_rooms_s_goals and k != dict_game_goals['cooking_location']:
                    clue = clue + 'the room {}, '.format(dict_rooms_numbers[k])
                    uninteresting_place = True
            if uninteresting_place:
                clue = clue[:-2] + ' are uninteresting places'
        else:
            clue = clue + 'there is only one room'
        clue = clue + '\r\n'
        f.write(clue)
        f.write('\n')

    if name_type == 'room_importance':
        f.write('\n')

        f.write('easy:\r')
        clue = 'Go in rooms with a 2, finish by the room with 1'
        if death_room is not None:
            clue += ', avoid room -1'
        f.write(clue)
        f.write('\r \n')

        f.write('medium:\r')
        clue = 'rooms with 0 have no interest;  room with 1 is the place of the main quest; rooms with 2 are the ' \
               'places for secondary quest'
        if death_room is not None:
            clue += ' and avoid room with -1 which is the death room'
        f.write(clue)
        f.write('\r \n')

        f.write('hard:\r')
        clue = 'Go in rooms with a 2 finish by the room with 1'
        f.write(clue)
        f.write('\r \n')

        f.write('very hard:\r')
        clue = 'rooms with a 2 are less important than room with 1 and rooms with 0 have no importance'
        f.write(clue)

    f.write('\r \n')
    f.write('####end clue###\r')

    rooms_leading_to_death_room = dict()
    if death_room is not None:
        f.write('\r \n')
        f.write('####death room###\r')

        for i in range(4):
            if rooms_dict[death_room][i] is not None:
                if i == 0:
                    rooms_leading_to_death_room['south'] = rooms_dict[death_room][i]
                elif i == 1:
                    rooms_leading_to_death_room['north'] = rooms_dict[death_room][i]
                elif i == 2:
                    rooms_leading_to_death_room['west'] = rooms_dict[death_room][i]
                else:
                    rooms_leading_to_death_room['east'] = rooms_dict[death_room][i]

        f.write('\n')

        f.write('easy:\r')
        clue = 'the {} is the death room,'.format(death_room)
        for k in iter(rooms_leading_to_death_room):
            clue = clue + 'the {} go to death room by {}'.format(rooms_leading_to_death_room[k], k) + ' '
        f.write(clue)
        f.write('\r \n')

        f.write('medium:\r')
        clue = ''
        for k in iter(rooms_leading_to_death_room):
            clue = clue + 'the {} go to death room by {}'.format(rooms_leading_to_death_room[k], k) + ' '
        f.write(clue)
        f.write('\r \n')

        f.write('hard:\r')
        clue = ''
        for k in iter(rooms_leading_to_death_room):
            clue = clue + 'avoid the {} of {}'.format(k, rooms_leading_to_death_room[k]) + ' '
        f.write(clue)
        f.write('\r \n')

        f.write('very hard:\r')
        clue = 'from the death room there is '
        for i in range(4):
            if rooms_dict[death_room][i] is not None:
                if i == 0:
                    clue = clue + 'the {} on the {},'.format(rooms_dict[death_room][i], 'north')
                elif i == 1:
                    clue = clue + 'the {} on the {},'.format(rooms_dict[death_room][i], 'south')
                elif i == 2:
                    clue = clue + 'the {} on the {},'.format(rooms_dict[death_room][i], 'east')
                else:
                    clue = clue + 'the {} on the {},'.format(rooms_dict[death_room][i], 'west')

        f.write(clue)
        f.write('\r \n')
        f.write('####end death room###\r')

    f.close()
    return rooms_leading_to_death_room