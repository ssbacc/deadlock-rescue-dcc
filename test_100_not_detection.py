import os
import random
import pickle
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import Environment
from model import Network
import config
import copy
import json
import time

detection_interval = 4
resolution_interval = 4


# 함수들
directiondict = {
    'stay': 4, 'north': 0, 'south': 1, 'west': 2, 'east': 3
}
reverse_directiondict = {v: k for k, v in directiondict.items()}

def get_possible_directions(obs, obs_agents, agent_idx, agents_not_exchangeable, agents_fixed):
    directions = []
    directions_pushed_agents = []
    if obs[0][agent_idx][1][3, 4] == 0:
        directions.append('north')
    if obs[0][agent_idx][1][5, 4] == 0:
        directions.append('south')
    if obs[0][agent_idx][1][4, 3] == 0:
        directions.append('west')
    if obs[0][agent_idx][1][4, 5] == 0:
        directions.append('east')

    direction_conditions = [
        ('north', obs_agents[agent_idx][3, 4] - 1),
        ('south', obs_agents[agent_idx][5, 4] - 1),
        ('west', obs_agents[agent_idx][4, 3] - 1),
        ('east', obs_agents[agent_idx][4, 5] - 1)
    ]

    for direction, agent_value in direction_conditions:
        if agent_value in agents_not_exchangeable or agent_value in agents_fixed:
            if direction in directions:
                directions.remove(direction)

    for direction, agent_value in direction_conditions:
        if direction in directions:
            directions_pushed_agents.append((direction, None if agent_value == -1 else agent_value))

    return directions_pushed_agents


def get_possible_directions_super(obs, obs_agents, agent_idx, agents_fixed):
    directions = []
    directions_pushed_agents = []
    if obs[0][agent_idx][2][4, 4] == 1:
        directions.append('north')
    if obs[0][agent_idx][3][4, 4] == 1:
        directions.append('south')
    if obs[0][agent_idx][4][4, 4] == 1:
        directions.append('west')
    if obs[0][agent_idx][5][4, 4] == 1:
        directions.append('east')

    direction_conditions = [
        ('north', obs_agents[agent_idx][3, 4] - 1),
        ('south', obs_agents[agent_idx][5, 4] - 1),
        ('west', obs_agents[agent_idx][4, 3] - 1),
        ('east', obs_agents[agent_idx][4, 5] - 1)
    ]

    for direction, agent_value in direction_conditions:
        if agent_value in agents_fixed:
            if direction in directions:
                directions.remove(direction)

    for direction, agent_value in direction_conditions:
        if direction in directions:
            directions_pushed_agents.append((direction, None if agent_value == -1 else agent_value))

    return directions_pushed_agents


def push_recursive(obs, obs_agents, agent_super, agents_fixed):
    relayed_actions = []
    agents_not_exchangeable = []

    current_agent = agent_super
    depth = 0

    # 스택에 (현재 에이전트, 남은 방향들, depth) 저장
    stack = []

    while True:
        # 가능한 방향들 계산
        if current_agent == agent_super:
            possible_directions = get_possible_directions_super(obs, obs_agents, current_agent, agents_fixed)
        else:
            possible_directions = get_possible_directions(obs, obs_agents, current_agent, agents_not_exchangeable, agents_fixed)

        while not possible_directions:
            if not stack:
                # 백트래킹할 곳이 없으면 종료
                return [(agent_super, 'stay')]

            # 스택에서 이전 상태로 백트래킹
            last_agent, last_possible_directions, last_depth = stack.pop()

            # 남은 방향이 있다면 그 중 하나를 선택하고 진행
            if last_possible_directions:
                relayed_actions = relayed_actions[:last_depth]  # 이전 선택을 지우고 다시 선택
                current_agent = last_agent
                possible_directions = last_possible_directions
                depth = last_depth
            else:
                # 백트래킹할 방향이 없으면 계속 백트래킹
                possible_directions = []

        # 랜덤으로 가능한 방향 중 하나 선택
        choosen_action = random.choice(possible_directions)
        possible_directions.remove(choosen_action)

        relayed_actions.append((current_agent, choosen_action[0]))

        # 더 이상 밀 에이전트가 없으면 종료
        if choosen_action[1] is None:
            break

        # depth에 따른 agents_not_exchangeable 처리
        if depth == 1:
            agents_not_exchangeable = []
        agents_not_exchangeable.append(current_agent)

        # 스택에 현재 상태를 저장
        stack.append((current_agent, possible_directions, depth))

        # 다음 에이전트를 선택하고 루프를 계속
        current_agent = choosen_action[1]
        depth += 1

    return relayed_actions


def get_possible_directions_radiation(obs, obs_agents, center_coordinates, agent_idx, agents_fixed):
    directions = []
    directions_pushed_agents = []
    if obs[0][agent_idx][1][3, 4] == 0:
        directions.append('north')
    if obs[0][agent_idx][1][5, 4] == 0:
        directions.append('south')
    if obs[0][agent_idx][1][4, 3] == 0:
        directions.append('west')
    if obs[0][agent_idx][1][4, 5] == 0:
        directions.append('east')
    
    row_diff = center_coordinates[0] - obs[2][agent_idx][0]
    col_diff = center_coordinates[1] - obs[2][agent_idx][1]

    if row_diff < 0:  # 에이전트가 중앙보다 아래에 있으면 북쪽으로 이동 불가
        if 'north' in directions:
            directions.remove('north')
    elif row_diff > 0:  # 에이전트가 중앙보다 위에 있으면 남쪽으로 이동 불가
        if 'south' in directions:
            directions.remove('south')
    if col_diff < 0:  # 에이전트가 중앙보다 오른쪽에 있으면 서쪽으로 이동 불가
        if 'west' in directions:
            directions.remove('west')
    elif col_diff > 0:  # 에이전트가 중앙보다 왼쪽에 있으면 동쪽으로 이동 불가
        if 'east' in directions:
            directions.remove('east')

    direction_conditions = [
        ('north', obs_agents[agent_idx][3, 4] - 1),
        ('south', obs_agents[agent_idx][5, 4] - 1),
        ('west', obs_agents[agent_idx][4, 3] - 1),
        ('east', obs_agents[agent_idx][4, 5] - 1)
    ]

    for direction, agent_value in direction_conditions:
        if agent_value in agents_fixed:
            if direction in directions:
                directions.remove(direction)

    for direction, agent_value in direction_conditions:
        if direction in directions:
            directions_pushed_agents.append((direction, None if agent_value == -1 else agent_value))

    return directions_pushed_agents


def push_recursive_radiation(obs, obs_agents, center_coordinates, agent_idx, agents_fixed):

    relayed_actions = []
    agents_not_exchangeable = []
    
    current_agent = agent_idx
    depth = 0

    # 스택에 (현재 에이전트, 남은 방향들, depth) 저장
    stack = []

    while True:
        # 가능한 방향들 계산
        if current_agent == agent_idx:
            possible_directions = get_possible_directions_radiation(obs, obs_agents, center_coordinates, current_agent, agents_fixed)
        else:
            possible_directions = get_possible_directions(obs, obs_agents, current_agent, agents_not_exchangeable, agents_fixed)

        while not possible_directions:
            if not stack:
                # 백트래킹할 곳이 없으면 종료'
                return [(agent_idx, 'stay')]

            # 스택에서 이전 상태로 백트래킹
            last_agent, last_possible_directions, last_depth = stack.pop()

            # 남은 방향이 있다면 그 중 하나를 선택하고 진행
            if last_possible_directions:
                relayed_actions = relayed_actions[:last_depth]  # 이전 선택을 지우고 다시 선택
                current_agent = last_agent
                possible_directions = last_possible_directions
                depth = last_depth
            else:
                # 백트래킹할 방향이 없으면 계속 백트래킹
                possible_directions = []

        # 랜덤으로 가능한 방향 중 하나 선택
        choosen_action = random.choice(possible_directions)
        possible_directions.remove(choosen_action)

        relayed_actions.append((current_agent, choosen_action[0]))

        # 더 이상 밀 에이전트가 없으면 종료
        if choosen_action[1] is None:
            break

        # depth에 따른 agents_not_exchangeable 처리
        if depth == 1:
            agents_not_exchangeable = []
        agents_not_exchangeable.append(current_agent)

        # 스택에 현재 상태를 저장
        stack.append((current_agent, possible_directions, depth))

        # 다음 에이전트를 선택하고 루프를 계속
        current_agent = choosen_action[1]
        depth += 1

    return relayed_actions


def get_possible_directions_not_deadlock(obs, obs_agents, agent_idx, agent_action, agents_fixed):
    directions = [agent_action]
    directions_pushed_agents = []

    direction_conditions = [
        ('north', obs_agents[agent_idx][3, 4] - 1),
        ('south', obs_agents[agent_idx][5, 4] - 1),
        ('west', obs_agents[agent_idx][4, 3] - 1),
        ('east', obs_agents[agent_idx][4, 5] - 1)
    ]

    for direction, agent_value in direction_conditions:
        if agent_value in agents_fixed:
            if direction in directions:
                directions.remove(direction)

    for direction, agent_value in direction_conditions:
        if direction in directions:
            directions_pushed_agents.append((direction, None if agent_value == -1 else agent_value))

    return directions_pushed_agents


def push_recursive_not_deadlock(obs, obs_agents, agent_idx, agent_action, agents_fixed):

    relayed_actions = []
    agents_not_exchangeable = []
    
    current_agent = agent_idx
    depth = 0

    # 스택에 (현재 에이전트, 남은 방향들, depth) 저장
    stack = []

    while True:
        # 가능한 방향들 계산
        if current_agent == agent_idx:
            possible_directions = get_possible_directions_not_deadlock(obs, obs_agents, agent_idx, agent_action, agents_fixed)
        else:
            possible_directions = get_possible_directions(obs, obs_agents, current_agent, agents_not_exchangeable, agents_fixed)

        while not possible_directions:
            if not stack:
                # 백트래킹할 곳이 없으면 종료'
                return [(agent_idx, 'stay')]

            # 스택에서 이전 상태로 백트래킹
            last_agent, last_possible_directions, last_depth = stack.pop()

            # 남은 방향이 있다면 그 중 하나를 선택하고 진행
            if last_possible_directions:
                relayed_actions = relayed_actions[:last_depth]  # 이전 선택을 지우고 다시 선택
                current_agent = last_agent
                possible_directions = last_possible_directions
                depth = last_depth
            else:
                # 백트래킹할 방향이 없으면 계속 백트래킹
                possible_directions = []

        # 랜덤으로 가능한 방향 중 하나 선택
        choosen_action = random.choice(possible_directions)
        possible_directions.remove(choosen_action)

        relayed_actions.append((current_agent, choosen_action[0]))

        # 더 이상 밀 에이전트가 없으면 종료
        if choosen_action[1] is None:
            break

        # depth에 따른 agents_not_exchangeable 처리
        if depth == 1:
            agents_not_exchangeable = []
        agents_not_exchangeable.append(current_agent)

        # 스택에 현재 상태를 저장
        stack.append((current_agent, possible_directions, depth))

        # 다음 에이전트를 선택하고 루프를 계속
        current_agent = choosen_action[1]
        depth += 1

    return relayed_actions


def get_sorted_agents_super(agent_groups, env):
    super_agents = []
    for set_of_agents in agent_groups:
        if not set_of_agents:
            continue
        agent_super = max(set_of_agents, key=lambda i: np.sum(np.abs(env.agents_pos[i] - env.goals_pos[i])))
        super_agents.append(agent_super)

    if not super_agents:
        return []

    # 각 에이전트와 목표 사이의 거리 계산
    agent_distances = [(agent, np.sum(np.abs(env.agents_pos[agent] - env.goals_pos[agent]))) for agent in super_agents]

    # 거리를 기준으로 내림차순 정렬
    sorted_agents = sorted(agent_distances, key=lambda x: x[1], reverse=True)

    # 정렬된 에이전트 ID 추출
    sorted_agent_groups = [agent for agent, distance in sorted_agents]
    return sorted_agent_groups


def get_sorted_agents(agent_groups, env):
    # 각 에이전트와 목표 사이의 거리 계산
    agent_distances = [(agent, np.sum(np.abs(env.agents_pos[agent] - env.goals_pos[agent]))) for agent in agent_groups]

    # 거리를 기준으로 내림차순 정렬
    sorted_agents = sorted(agent_distances, key=lambda x: x[1], reverse=True)

    # 정렬된 에이전트 ID 추출
    sorted_agent_groups = [agent for agent, distance in sorted_agents]
    return sorted_agent_groups

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
torch.set_num_threads(1)


def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set_100/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)


def code_test():
    env = Environment()
    network = Network()
    network.eval()
    obs, last_act, pos = env.observe()
    network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
    

def test_model(model_range: Union[int, tuple], test_set=config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2)

    if isinstance(model_range, int):
        state_dict = torch.load(os.path.join(config.save_path, f'{model_range}.pth'), map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print(f'----------test model {model_range}----------')

        instance_id = 0

        for case in test_set:
            print(f"test set: {case[0]} env {case[1]} agents")
            with open('./test_set_100/{}_{}agents.pth'.format(case[0], case[1]), 'rb') as f:
                tests = pickle.load(f)

            # test = tests[0]
            # ret = test_one_case((test, network, instance_id))

            # success, steps, num_comm = ret

            instance_id_base = instance_id
            tests = [(test, network, instance_id_base + i) for i, test in enumerate(tests)]
            ret = pool.map(test_one_case, tests)

            success, steps, num_comm = zip(*ret)

            print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
            print("average step: {}".format(sum(steps)/len(steps)))
            print("communication times: {}".format(sum(num_comm)/len(num_comm)))
            print()

            instance_id += len(tests)

    elif isinstance(model_range, tuple):

        for model_name in range(model_range[0], model_range[1]+1, config.save_interval):
            state_dict = torch.load(os.path.join(config.save_path, f'{model_name}.pth'), map_location=DEVICE)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()


            print(f'----------test model {model_name}----------')

            instance_id = 0

            for case in test_set:
                print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
                with open(f'./test_set_100/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
                    tests = pickle.load(f)

                # test = tests[0]
                # ret = test_one_case((test, network, instance_id))

                # success, steps, num_comm = ret

                instance_id_base = instance_id
                tests = [(test, network, instance_id_base + i) for i, test in enumerate(tests)]
                ret = pool.map(test_one_case, tests)

                success, steps, num_comm = zip(*ret)

                print("success rate: {:.2f}%".format(sum(success)/len(success)*100))
                print("average step: {}".format(sum(steps)/len(steps)))
                print("communication times: {}".format(sum(num_comm)/len(num_comm)))
                print()

                instance_id += 1

            print('\n')


def test_one_case(args):

    env_set, network, instance_id = args

    env = Environment()
    env.load(np.array(env_set[0]), np.array(env_set[1]), np.array(env_set[2]))
    obs, last_act, pos = env.observe()
    
    done = False
    network.reset()

    num_agents = len(env_set[1])

    step = 0
    num_comm = 0

    while not done and env.steps < config.max_episode_length:
        env_copy = copy.deepcopy(env)
        plan = []
        not_arrived = set()
        sim_obs, sim_last_act, sim_pos = env_copy.observe()
        sim_done = False
        for _ in range(detection_interval):
            if env_copy.steps >= config.max_episode_length or sim_done:
                break
            actions, _, _, _, comm_mask = network.step(torch.as_tensor(sim_obs.astype(np.float32)).to(DEVICE), 
                                                        torch.as_tensor(sim_last_act.astype(np.float32)).to(DEVICE), 
                                                        torch.as_tensor(sim_pos.astype(int)))
            plan.append((actions, comm_mask, copy.deepcopy(sim_pos)))
            (sim_obs, sim_last_act, sim_pos), _, sim_done, _ = env_copy.step(actions)
            for i in range(num_agents):
                if not np.array_equal(env_copy.agents_pos[i], env_copy.goals_pos[i]):
                    not_arrived.add(i)
        not_arrived = list(not_arrived)

        deadlock_exists = len(not_arrived) > 0
        
        if not deadlock_exists:
            for actions, comm_mask, _ in plan:
                if env.steps >= config.max_episode_length or done:
                    break
                (obs, last_act, pos), _, done, _ = env.step(actions)
                # env.save_frame(step, instance_id)
                step += 1
                num_comm += np.sum(comm_mask)
        else:
            sorted_leader_agents = get_sorted_agents(not_arrived, env)

            for _ in range(resolution_interval):
                if env.steps >= config.max_episode_length or done:
                    break
                obs_agents = env.observe_agents()
                observation = env.observe()

                manual_actions = [4 for _ in range(num_agents)]
                
                fixed_agents = []
                for super_agent in sorted_leader_agents:
                    if super_agent in fixed_agents:
                        continue
                    for relayed_action in push_recursive(observation, obs_agents, super_agent, fixed_agents):
                        manual_actions[relayed_action[0]] = directiondict[relayed_action[1]]
                        fixed_agents.append(relayed_action[0])

                (obs, last_act, pos), _, done, _ = env.step(manual_actions)
                # env.save_frame(step, instance_id)
                step += 1
                num_comm += np.sum(comm_mask)

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm


if __name__ == '__main__':
    start_time = time.time()
    test_model(128000)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")