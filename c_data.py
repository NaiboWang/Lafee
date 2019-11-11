import json, os, re, time, threading, traceback

# Global variables
PATH = {'raw': '../../fb_dataset/fb_player_event_json/dataset/',
        'input': '../../fb_dataset/fb_player_event_json/dataset_input/',
        'washed_json': '../../fb_dataset/fb_player_event_json/dataset_washed_json/'}
FILES = [i for i in os.listdir(PATH['raw']) if re.match(r'.*json', i)]
STATISTIC_DICT = {'total': 0}
ACTION_LIST = {'LoginRole', 'LogoutRole', 'ReplaceRole', 'PrivateGame', 'QuickMatch1V1', 'QuickMatch2V2',
               'DailyTaskFinish', 'DailyTaskReward', 'DailySign', 'DailySignReward', 'RewardAchievement', 'InviteLog',
               'ShareLog', 'FollowLog', 'PraisePlayRound', 'RoomModeCreate', 'ConsumeItem', 'GuideInfo', 'AdsLog'}
STATE_LIST = {'Trade', 'AddExp', 'SendEmotion', 'SendGift', 'AddAchievement', 'backpack', 'GradeUp', 'Duration'}


# Split items and run function through n threads
def run_through_threads(func, arg_list, items, num=4):
    threads = []
    item_len = len(items)
    for i in range(num):
        threads.append(threading.Thread(target=func,
                                        args=(*arg_list, items[int(i * item_len / num):int((i + 1) * item_len / num)])))
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()


# Count the number of each log
def count_log_number(src, dst):
    print('Counting the number of each log... ')
    for FILE in FILES:
        print(FILE)
        with open(src + FILE, 'r') as file:
            logs = json.load(file)
            for log in logs:
                # # Count Trade reason
                # if log['log_id'] == 'Trade':
                #     try:
                #         STATISTIC_DICT[log['raw_info']['reason']] += 1
                #     except:
                #         STATISTIC_DICT[log['raw_info']['reason']] = 1
                try:
                    STATISTIC_DICT[log['log_id']] += 1
                except:
                    STATISTIC_DICT[log['log_id']] = 1
                finally:
                    STATISTIC_DICT['total'] += 1
    STATISTIC_DICT_SORTED = sorted(STATISTIC_DICT.items(), key=lambda d: d[1], reverse=True)
    STATISTIC_DICT2W = str(STATISTIC_DICT_SORTED)[2: -2].replace('), (', '\n')
    with open(dst + 'statistic.csv', 'w') as file:
        file.write(STATISTIC_DICT2W)
    print('Finished')


# Fix the bugs in the data
def fix_data_bug(src, dst, items):
    print('Fixing the bugs in the data... ')
    unusable_keys = ['QuickMatch2V2', 'QuickMatch1V1']
    for FILE in items:
        with open(src + FILE, 'r') as file:
            try:
                logs = json.load(file)
                # Put the actions before the state which were logged at the same time
                logs_ordered = []
                state_count = 0
                timestamp = logs[0]['timestamp']
                for pointer, _ in enumerate(logs):
                    try:
                        while not logs[pointer].__contains__("raw_info"):
                            print(logs[pointer])
                            logs.pop(pointer)
                    except:
                        pass
                for pointer, log in enumerate(logs):
                    if log['log_id'] in STATE_LIST:
                        logs_ordered.append(log)
                        if log['timestamp'] == timestamp:
                            state_count += 1
                        else:
                            timestamp = log['timestamp']
                            state_count = 1
                    elif log['log_id'] in ACTION_LIST or log['log_id'] == 'MatchInfo':
                        if log['timestamp'] == timestamp:
                            if state_count != 0:
                                logs_ordered.insert(-state_count, log)
                            else:
                                logs_ordered.append(log)
                        else:
                            logs_ordered.append(log)
                            timestamp = log['timestamp']
                            state_count = 0
                    else:
                        print('\033[1;35m', log['log_id'], 'is included in neither action nor state list!', '\033[0m')
                logs = logs_ordered
                logs.reverse()
                # Turn wrong LoginRole into PrivateGame
                # Fix the login order
                for pointer, _ in enumerate(logs):
                    try:
                        if logs[pointer]['log_id'] == 'LoginRole':
                            if 'ticket' in logs[pointer]['raw_info'].keys():
                                logs[pointer]['log_id'] = 'PrivateGame'
                            elif (pointer + 1) < len(logs) and logs[pointer + 1]['timestamp'] == logs[pointer]['timestamp']:
                                temp = logs[pointer]
                                logs[pointer] = logs[pointer + 1]
                                logs[pointer + 1] = temp
                    except Exception as e:
                        traceback.print_exc()
                logs.reverse()
                # Fix the logout order
                for pointer, log in enumerate(logs):
                    if log['log_id'] == 'LogoutRole':
                        if (pointer + 1) < len(logs) and logs[pointer + 1]['timestamp'] == log['timestamp']:
                            logs[pointer] = logs[pointer + 1]
                            logs[pointer + 1] = log
                # Delete unusable keys
                # Turn MatchInfo into 1v1 2v2
                # Delete unusable message
                for pointer, _ in enumerate(logs):
                    while logs[pointer]['log_id'] in unusable_keys:
                        logs.pop(pointer)
                        if pointer >= len(logs):
                            break
                    if pointer >= len(logs):
                        break
                    if logs[pointer]['log_id'] == 'MatchInfo':
                        try:
                            if logs[pointer]['raw_info']['matchmodeid'] == 1001:
                                logs[pointer]['log_id'] = 'QuickMatch1V1'
                            elif logs[pointer]['raw_info']['matchmodeid'] == 1002:
                                logs[pointer]['log_id'] = 'QuickMatch2V2'
                        except:
                            print('\033[1;35m', FILE, logs[pointer], 'matchmodeid unconsidered','\033[0m')
                    elif logs[pointer]['log_id'] == 'Trade':
                        logs[pointer]['raw_info'] = logs[pointer]['raw_info']['gain_count']
                    elif logs[pointer]['log_id'] == 'AddExp':
                        logs[pointer]['raw_info'] = logs[pointer]['raw_info']['changevalue']
                    elif logs[pointer]['log_id'] == 'SendEmotion':
                        logs[pointer]['raw_info'] = 1
                    elif logs[pointer]['log_id'] == 'SendGift':
                        logs[pointer]['raw_info'] = 1
                    elif logs[pointer]['log_id'] == 'GradeUp':
                        logs[pointer]['raw_info'] = 1
                    elif logs[pointer]['log_id'] == 'backpack':
                        logs[pointer]['raw_info'] = logs[pointer]['raw_info']['ItemCount']
                    elif logs[pointer]['log_id'] == 'ConsumeItem':
                        logs[pointer]['raw_info'] = logs[pointer]['raw_info']['ItemCount']
                    elif logs[pointer]['log_id'] == 'AddAchievement':
                        logs[pointer]['raw_info'] = 1

            except Exception as e:
                traceback.print_exc()
                print('\033[1;35m', FILE, log['log_id'], pointer, '\033[0m')
        with open(dst + FILE, 'w') as file:
            json.dump(logs, file)
    print('Finished')


# Reset state
def reset_dict(dict, reset_value):
    for key in list(dict.keys()):
        dict[key] = reset_value


# Calculate time interval (absolute)
def cal_time_interval_abs(time1, time2):
    return abs(int(time.mktime(time.strptime(time1, "%Y-%m-%d %H:%M:%S"))) - int(
        time.mktime(time.strptime(time2, "%Y-%m-%d %H:%M:%S"))))


# Get the usable data for ML
def generate_input_data(src, dst, items):
    print('Generating data for ML process...')
    for FILE in items:
        with open(src + FILE, 'r') as file, open(dst + FILE[:-4] + 'csv', 'w') as file2:
            states = {'Trade': 0,
                      'AddExp': 0,
                      'SendEmotion': 0,
                      'SendGift': 0,
                      'AddAchievement': 0,
                      'backpack': 0,
                      'GradeUp': 0,
                      'Duration': 0, }
            actions = {'LoginRole': 0,
                       'LogoutRole': 0,
                       'ReplaceRole': 0,
                       'PrivateGame': 0,
                       'QuickMatch1V1': 0,
                       'QuickMatch2V2': 0,
                       'DailyTaskFinish': 0,
                       'DailyTaskReward': 0,
                       'DailySign': 0,
                       'DailySignReward': 0,
                       'RewardAchievement': 0,
                       'InviteLog': 0,
                       'ShareLog': 0,
                       'FollowLog': 0,
                       'PraisePlayRound': 0,
                       'RoomModeCreate': 0,
                       'ConsumeItem': 0,
                       'GuideInfo': 0,
                       'AdsLog': 0, }
            states_l = states.copy()
            time = 0
            try:
                # file2.write(str([*list(states.keys()),*list(actions.keys()), 'time'])[1:-1] + '\n') # add header of csv
                logs = json.load(file)
                timestamp = logs[0]['timestamp']
                for pointer, log in enumerate(logs):
                    if log['log_id'] in states.keys():
                        states[log['log_id']] += log['raw_info']
                    if log['log_id'] in actions.keys():
                        if timestamp == log['timestamp']:
                            actions[log['log_id']] = 1
                        else:
                            time = cal_time_interval_abs(log['timestamp'], timestamp)
                            timestamp = log['timestamp']
                            states['Duration'] += time
                            file2.write(str([*list(states_l.values()), *list(actions.values()), time])[1:-1] + '\n')
                            states_l = states.copy()
                            if log['log_id'] == 'LoginRole':
                                reset_dict(states, 0)
                            reset_dict(actions, 0)
                            actions[log['log_id']] = 1
                        if log['log_id'] == 'ConsumeItem':
                            states['backpack'] -= log['raw_info']

                    # if log['log_id'] == 'LogoutRole':
                    #     reset_dict(actions, 0)
                    #     actions[log['log_id']] = 1
                    #     states['Duration'] = states['Duration'] + cal_time_interval_abs(log['timestamp'], timestamp)
                    #     if logs[pointer+1]['log_id'] == 'LoginRole':
                    #         time = cal_time_interval_abs(logs[pointer+1]['timestamp'], log['timestamp'])
                    # elif log['log_id'] == 'LoginRole':
                    #     reset_dict(states, 0)
                    #     actions[log['log_id']] = 1

            except Exception as e:
                traceback.print_exc()
                print('\033[1;35m', FILE, log['log_id'], pointer, '\033[0m')
    print('Finished')


if __name__ == '__main__':
    # todo: trade reason与行为log的对应关系
    # fix_data_bug(PATH['raw'], PATH['washed_json'], FILES)
    # run_through_threads(fix_data_bug, [PATH['raw'], PATH['washed_json']], FILES, 10)
    # count_log_number(PATH['washed_json'], PATH['input'])
    # generate_input_data(PATH['washed_json'], PATH['input'], FILES)
    run_through_threads(generate_input_data, [PATH['washed_json'], PATH['input']], FILES, 10)
