import pandas as pd
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from env.StockTradingEnv import StockTradingEnv
from util import find_file, plot_daily_profits

def stock_trade(stock_file, RL_model):

    # 模型训练
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    if RL_model == 'A2C':
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'DDPG':
        model = DDPG("MlpPolicy", env, verbose=0, tensorboard_log='./log')
    elif RL_model == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, tensorboard_log='./log')

    model.learn(total_timesteps=int(1e4))

    # 模型测试
    day_profits = []
    df_test = pd.read_csv(stock_file.replace('train', 'test'))
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits

def test_a_stock_trade(stock_code, RL_model):

    stock_file = find_file('./data/train', str(stock_code))
    daily_profits = stock_trade(stock_file, RL_model)
    plot_daily_profits(stock_code, RL_model, daily_profits)

if __name__ == '__main__':
    test_a_stock_trade('sh.600000', 'PPO')


