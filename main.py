import pandas as pd
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from env.StockTradingEnv import StockTradingEnv
from util import find_file, plot_daily_profits
import yaml

with open('config.yaml') as f:
    args = yaml.safe_load(f)

def prepare_env(stock_file):
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    return env, len(df)

def train_model(env, RL_model):
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

def test_model(test_env, len_test, model):
    dates = []
    daily_profits = []
    daily_opens = []
    daily_closes = []
    daily_highs = []
    daily_lows = []
    obs = test_env.reset()
    for i in range(len_test - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        date, profit, open, close, high, low = test_env.render()
        dates.append(date)
        daily_profits.append(profit)
        daily_opens.append(open)
        daily_closes.append(close)
        daily_highs.append(high)
        daily_lows.append(low)
        if done:
            break
    return dates, daily_profits, daily_opens, daily_closes, daily_highs, daily_lows

def train_and_test_strategy(stock_code, RL_model):

    stock_file = find_file('./data/train', str(stock_code))
    train_env, _ = prepare_env(stock_file)
    model = train_model(train_env, RL_model)

    stock_file = find_file('./data/test', str(stock_code))
    test_env, len_test = prepare_env(stock_file)

    dates, daily_profits, daily_opens, daily_closes, daily_highs, daily_lows = test_model(test_env, len_test, model)
    plot_daily_profits(stock_code, RL_model, daily_profits, dates, daily_opens, daily_closes, daily_highs, daily_lows)

if __name__ == '__main__':

    train_and_test_strategy(args['train_args']['stock_code'], args['train_args']['rl_model'])


