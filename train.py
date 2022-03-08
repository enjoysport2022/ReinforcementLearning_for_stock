import os
import yaml
from stable_baselines3 import PPO, A2C, DDPG, TD3
from util import find_file, prepare_env
# from autox import AutoX

with open('config.yaml') as f:
    args = yaml.safe_load(f)

def train_model(env, RL_model='PPO'):
    if RL_model == 'A2C':
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'DDPG':
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log='./log')

    model.learn(total_timesteps=args['train_args']['total_timesteps'])

    return model

def train_strategy(stock_code, RL_model):

    stock_file = find_file('./data/tushare_data/train', str(stock_code))
    train_env, _ = prepare_env(stock_file)
    model = train_model(train_env, RL_model)

    os.makedirs('./check_points/', exist_ok=True)
    model.save(f"./check_points/{RL_model}_{stock_code}")

if __name__ == '__main__':

    train_strategy(args['train_args']['stock_code'], args['train_args']['rl_model'])


