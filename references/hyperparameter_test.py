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
from openai import OpenAI
import json
import time

detection_interval = 4
resolution_interval = 16
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()


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


def get_sorted_agents_no_deadlock(agent_groups, env):
    # 각 에이전트와 목표 사이의 거리 계산
    agent_distances = [(agent, np.sum(np.abs(env.agents_pos[agent] - env.goals_pos[agent]))) for agent in agent_groups]

    # 거리를 기준으로 내림차순 정렬
    sorted_agents = sorted(agent_distances, key=lambda x: x[1], reverse=True)

    # 정렬된 에이전트 ID 추출
    sorted_agent_groups = [agent for agent, distance in sorted_agents]
    return sorted_agent_groups


# 프롬프트
class gpt4pathfinding:
    def detection(self, agents_state):
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are the manager responsible for detecting whether agents are deadlocked in a MAPF problem. You can infer each agent's state based on their behavior."},
            {"role": "user", "content":
                f"""
                You are given {detection_interval} action logs of agents to detect deadlocks.
                
                Follow these steps in order:

                1. **Classify deadlocks**:
                    - Detect agents that are exhibiting deadlock conditions.
                    - Deadlock conditions:
                        - No movement while in the "Not arrived" state.
                        - Wandering behavior with no meaningful coordinate change during {detection_interval} behaviors in the "Not arrived" state.
                    - Not considered deadlocks:
                        - Transition from "Not arrived" to "Arrived" and remaining stationary.
                        - Remaining in the "Not arrived" state but showing consistent coordinate changes.

                2. **Group deadlocked agents**:
                    - Group deadlocked agents that are within a 2-Manhattan distance of each other. 2-Manhattan distance means that the sum of the absolute differences between the x-coordinates and y-coordinates of two agents is 2 or less.
                    - If a deadlocked agent is within a 2-Manhattan distance of another agent that has already arrived, include them in the same group, as these agents can still cause or experience deadlocks.

                3. **Provide solutions**:
                    - Use the "leader" method for independently deadlocked agents or if any agent in the group has a goal more than 8 units away in Manhattan distance.
                    - Use the "radiation" method if all agents in the group are close to their goals (less than 8 units), are deadlocked due to nearby agents, and are likely to experience repeated deadlocks.
                    - When a deadlocked agent has nearby agents that have already arrived, closely check the goals and apply the radiation method if necessary, as this is a key performance bottleneck.

                Rules:
                - Return "[]" if no deadlocks are found.
                - Ensure no duplicate agents.
                - Penalties apply for trivial or non-deadlock cases.
                
                Below are the {detection_interval} action logs of agents.

                {agents_state}

                Do not generate a description or explanation.

                Provide the agent group status in this JSON format:
                {{
                    "agent_id": [Agent IDs in the same group],
                    "solution": "leader" or "radiation"
                }}

                EXAMPLE 1:
                [
                    {{"agent_id": [1, 24], "solution": "leader"}},
                    {{"agent_id": [4, 5], "solution": "radiation"}}
                ]

                EXAMPLE 2:
                []

                EXAMPLE 3:
                [
                    {{"agent_id": [8], "solution": "leader"}}
                ]

                AGAIN, DO NOT GENERATE A DESCRIPTION OR EXPLANATION.
                """
            }],
        )
        return response.choices[0].message.content
    
pathfinder = gpt4pathfinding()


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

        observations = env.observe_agents()
        FOV_agents = [
            [*(observations[i][observations[i] != 0] - 1).tolist(), i] if np.any(observations[i]) else [i]
            for i in not_arrived
        ]
        agents_to_prompt = list({agent for agent_list in FOV_agents for agent in agent_list})

        planned_steps_dict = {i: [] for i in agents_to_prompt}
        goal_logged = {i: False for i in agents_to_prompt}
        for i in plan:
            actions, _, positions = i
            for agent_idx in agents_to_prompt:
                position = positions[agent_idx]
                # 목표 위치와 현재 위치를 비교하여 도달 여부 판단
                arrived_status = "Arrived" if np.array_equal(position, env.goals_pos[agent_idx]) else "Not arrived"
                direction = reverse_directiondict.get(actions[agent_idx], 'unknown')
                if not goal_logged[agent_idx]:
                    planned_steps_dict[agent_idx].append(
                        f"(Action: {direction}, Position: [{position[0]}, {position[1]}], {arrived_status})"
                    )
                    goal_logged[agent_idx] = True
                else:
                    planned_steps_dict[agent_idx].append(
                        f"(Action: {direction}, Position: [{position[0]}, {position[1]}], {arrived_status})"
                    )
        agents_state = ""
        for agent_idx in planned_steps_dict:
            agent_goal = f" (Goal: [{env.goals_pos[agent_idx][0]}, {env.goals_pos[agent_idx][1]}])"
            agent_log = ", ".join(planned_steps_dict[agent_idx])
            agents_state += f"Agent {agent_idx}{agent_goal}: {agent_log}\n"
        gpt4_response = pathfinder.detection(agents_state)
        response_text = gpt4_response
        try:
            start_idx = response_text.index('[')
            end_idx = response_text.rindex(']') + 1
            json_part = response_text[start_idx:end_idx]
            json_data = json.loads(json_part)
            # print("Extracted JSON:", json_data)
        except:
            # print("JSON 부분을 찾을 수 없으므로 deadlock이 없다고 가정합니다.")
            json_data = []

        deadlock_exists = len(json_data) > 0
        
        if not deadlock_exists:
            for actions, comm_mask, _ in plan:
                if env.steps >= config.max_episode_length or done:
                    break
                (obs, last_act, pos), _, done, _ = env.step(actions)
                # env.save_frame(step, instance_id)
                step += 1
                num_comm += np.sum(comm_mask)
        else:
            leader_agents = []
            radiation_agents = []

            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict):
                        if item.get('solution') == 'leader':
                            leader_agents.append(item['agent_id'])
                        elif item.get('solution') == 'radiation':
                            radiation_agents.append(item['agent_id'])

            leader_agents = [[agent for agent in group if agent < num_agents] for group in leader_agents]
            radiation_agents = [[agent for agent in group if agent < num_agents] for group in radiation_agents]

            deadlocked_agents = set()
            for group in leader_agents + radiation_agents:
                deadlocked_agents.update(group)

            all_agents = set(range(num_agents))
            no_deadlock_agents = list(all_agents - deadlocked_agents)

            sorted_leader_agents = get_sorted_agents_super(leader_agents, env)
            sorted_no_deadlock_agents = get_sorted_agents_no_deadlock(no_deadlock_agents, env)

            for _ in range(resolution_interval):
                if env.steps >= config.max_episode_length or done:
                    break
                obs_agents = env.observe_agents()
                observation = env.observe()

                manual_actions = [4 for _ in range(num_agents)]
                ml_planned_actions, _, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE), 
                                                    torch.as_tensor(pos.astype(int)))
                
                fixed_agents = []
                for super_agent in sorted_leader_agents:
                    if super_agent in fixed_agents:
                        continue
                    for relayed_action in push_recursive(observation, obs_agents, super_agent, fixed_agents):
                        manual_actions[relayed_action[0]] = directiondict[relayed_action[1]]
                        fixed_agents.append(relayed_action[0])

                random.shuffle(radiation_agents)
                for set_of_agents in radiation_agents:
                    x_values = []
                    y_values = []
                    random.shuffle(set_of_agents)
                    for agent_idx in set_of_agents:
                        x_values.append(observation[2][agent_idx][0])
                        y_values.append(observation[2][agent_idx][1])
                    if len(x_values) == 0 or len(y_values) == 0:
                        continue
                    avg_x = sum(x_values) / len(x_values)
                    avg_y = sum(y_values) / len(y_values)
                    average_position = (avg_x, avg_y)

                    for radiation_agent in set_of_agents:
                        if radiation_agent in fixed_agents:
                            continue
                        for relayed_action in push_recursive_radiation(observation, obs_agents, average_position, radiation_agent, fixed_agents):
                            manual_actions[relayed_action[0]] = directiondict[relayed_action[1]]
                            fixed_agents.append(relayed_action[0])
                
                for no_deadlock_agent in sorted_no_deadlock_agents:
                    if no_deadlock_agent in fixed_agents:
                        continue
                    for relayed_action in push_recursive_not_deadlock(observation, obs_agents, no_deadlock_agent, reverse_directiondict[ml_planned_actions[no_deadlock_agent]], fixed_agents):
                        manual_actions[relayed_action[0]] = directiondict[relayed_action[1]]
                        fixed_agents.append(relayed_action[0])
                        
                (obs, last_act, pos), _, done, _ = env.step(manual_actions)
                # env.save_frame(step, instance_id)
                step += 1
                num_comm += np.sum(comm_mask)

    print('One instance completed')

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm


if __name__ == '__main__':
    start_time = time.time()
    test_model(128000)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")