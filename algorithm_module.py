# algorithm_module.py
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== 全局变量 ==========
Q = None
actions = None
states = None
is_trained = False

# ========== 动作空间 ==========
def get_light_actions():
    return [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 330, 360, 400]

# ========== 状态离散化 ==========
def discretize_state_combined(co2, temp, hum, light):
    # CO₂分级
    if co2 < 400:
        co2_level = 1
    elif co2 < 430:
        co2_level = 2
    elif co2 < 460:
        co2_level = 3
    elif co2 < 500:
        co2_level = 4
    elif co2 < 550:
        co2_level = 5
    elif co2 < 600:
        co2_level = 6
    elif co2 < 700:
        co2_level = 7
    elif co2 < 800:
        co2_level = 8
    elif co2 < 1000:
        co2_level = 9
    else:
        co2_level = 10
    
    # 温度分级
    if co2 > 600:
        if temp < 20:
            temp_level = 1
        elif temp < 24:
            temp_level = 2
        elif temp < 28:
            temp_level = 3
        elif temp < 32:
            temp_level = 4
        else:
            temp_level = 5
    else:
        if temp < 18:
            temp_level = 1
        elif temp < 22:
            temp_level = 2
        elif temp < 26:
            temp_level = 3
        elif temp < 30:
            temp_level = 4
        else:
            temp_level = 5
    
    # 湿度分级
    if hum < 40:
        hum_level = 1
    elif hum < 55:
        hum_level = 2
    elif hum < 70:
        hum_level = 3
    elif hum < 85:
        hum_level = 4
    else:
        hum_level = 5
    
    # 光照分级
    if light < 100:
        light_level = 1
    elif light < 160:
        light_level = 2
    elif light < 200:
        light_level = 3
    elif light < 240:
        light_level = 4
    elif light < 280:
        light_level = 5
    elif light < 330:
        light_level = 6
    elif light < 400:
        light_level = 7
    else:
        light_level = 8
    
    interaction_state = co2_level * 11 + temp_level
    return (interaction_state, hum_level, light_level)

# ========== 光合作用潜力计算 ==========
def calculate_photosynthesis_potential(co2, temp):
    if co2 < 350:
        co2_factor = 0.5
    elif co2 < 400:
        co2_factor = 0.7
    elif co2 < 450:
        co2_factor = 0.85
    elif co2 < 550:
        co2_factor = 1.0
    elif co2 < 700:
        co2_factor = 0.9
    elif co2 < 900:
        co2_factor = 0.75
    else:
        co2_factor = 0.5
    
    if co2 > 600:
        T_opt, T_min, T_max = 28, 18, 38
    elif co2 > 450:
        T_opt, T_min, T_max = 25, 15, 35
    else:
        T_opt, T_min, T_max = 22, 12, 32
    
    if temp < T_min:
        temp_factor = 0.3
    elif temp < T_opt - 5:
        temp_factor = 0.5 + (temp - (T_opt - 5)) * 0.1
    elif temp < T_opt + 5:
        temp_factor = 1.0 - abs(temp - T_opt) * 0.02
    elif temp < T_max:
        temp_factor = 0.6 - (temp - (T_opt + 5)) * 0.04
    else:
        temp_factor = 0.2
    
    interaction_modifier = 1.0 + (co2 - 400) / 1000 if co2 > 400 else 1.0
    interaction_modifier = min(interaction_modifier, 1.2)
    
    return min(max(co2_factor * temp_factor * interaction_modifier, 0.1), 1.0)

def get_optimal_light_range(potential):
    if potential > 0.85:
        return (280, 340), 310
    elif potential > 0.7:
        return (250, 300), 280
    elif potential > 0.55:
        return (220, 270), 250
    elif potential > 0.4:
        return (180, 230), 210
    elif potential > 0.25:
        return (140, 190), 170
    else:
        return (100, 150), 130

# ========== 奖励函数 ==========
def get_reward(light, co2, temp, hum, prev_light=None):
    reward = 0
    potential = calculate_photosynthesis_potential(co2, temp)
    optimal_range, optimal_light = get_optimal_light_range(potential)
    
    if co2 > 800:
        reward -= 0.8
    elif co2 > 700:
        reward -= 0.5
    elif co2 > 600:
        reward -= 0.3
    elif co2 > 550:
        reward -= 0.15
    elif co2 < 420:
        reward += 0.4
    elif co2 < 450:
        reward += 0.2
    
    if co2 > 600:
        if light > 300:
            reward -= 1.0
        elif light > 250:
            reward -= 0.5
        elif light < 180:
            reward += 0.3
    
    if co2 < 450:
        if light < 200:
            reward -= 0.6
        elif light < 250:
            reward -= 0.2
        elif light > 280:
            reward += 0.3
    
    if optimal_range[0] <= light <= optimal_range[1]:
        reward += 1.5
    elif abs(light - optimal_light) < 40:
        reward += 0.8
    elif abs(light - optimal_light) < 80:
        reward += 0.2
    else:
        reward -= 0.3
    
    if light > 400:
        reward -= 0.4
    elif light > 350 and potential < 0.6:
        reward -= 0.25
    
    if 40 <= hum <= 70:
        reward += 0.15
    elif hum > 85 or hum < 30:
        reward -= 0.1
    
    if prev_light is not None:
        change = abs(light - prev_light)
        if change > 100:
            reward -= 0.2
        elif change > 50:
            reward -= 0.1
    
    return reward

# ========== 训练函数 ==========
def train_algorithm(data_df, episodes=500, alpha=0.3, gamma=0.85, epsilon_start=0.5, epsilon_end=0.05):
    global Q, actions, states, is_trained
    
    actions = get_light_actions()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    
    states = []
    for i in range(len(data_df)):
        row = data_df.iloc[i]
        state = discretize_state_combined(
            row['CO2(ppm)'],
            row['温度(℃)'],
            row['湿度(%RH)'],
            row['光照(lx)']
        )
        states.append(state)
    
    train_rewards = []
    
    for episode in range(episodes):
        epsilon = epsilon_start * (epsilon_end / epsilon_start) ** (episode / episodes)
        prev_light = None
        episode_reward = 0
        
        for i in range(len(data_df) - 1):
            current_state = states[i]
            
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_values = Q[current_state]
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if q == max_q]
                action = random.choice(best_actions)
            
            next_row = data_df.iloc[i + 1]
            
            reward = get_reward(
                action,
                next_row['CO2(ppm)'],
                next_row['温度(℃)'],
                next_row['湿度(%RH)'],
                prev_light
            )
            
            episode_reward += reward
            next_state = states[i + 1]
            
            best_next_q = max(Q[next_state].values())
            Q[current_state][action] += alpha * (reward + gamma * best_next_q - Q[current_state][action])
            
            prev_light = action
        
        train_rewards.append(episode_reward)
    
    is_trained = True
    return Q, train_rewards

# ========== 推理函数 ==========
def predict_best_light(co2, temp, hum, current_light=0):
    global Q, is_trained
    
    if not is_trained or Q is None:
        raise ValueError("算法尚未训练，请先调用 train_algorithm()")
    
    state = discretize_state_combined(co2, temp, hum, current_light)
    
    if state not in Q:
        potential = calculate_photosynthesis_potential(co2, temp)
        _, optimal = get_optimal_light_range(potential)
        return optimal
    
    q_values = Q[state]
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    return int(np.mean(best_actions))

# ========== 模型保存和加载 ==========
def save_model(filepath="q_table.pkl"):
    global Q
    if Q is None:
        raise ValueError("没有可保存的Q表")
    with open(filepath, 'wb') as f:
        pickle.dump(dict(Q), f)
    return filepath

def load_model(filepath="q_table.pkl"):
    global Q, is_trained
    with open(filepath, 'rb') as f:
        Q = defaultdict(lambda: {a: 0.0 for a in get_light_actions()}, pickle.load(f))
    is_trained = True
    return Q
