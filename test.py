import yaml
from stable_baselines3 import PPO, A2C, DDPG, TD3
from util import find_file, plot_daily_profits, prepare_env

with open('config.yaml') as f:
    args = yaml.safe_load(f)

def load_model(RL_model, stock_code):

    if RL_model == 'A2C':
        model = A2C.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'PPO':
        model = PPO.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'DDPG':
        model = DDPG.load(f"./check_points/{RL_model}_{stock_code}")
    elif RL_model == 'TD3':
        model = TD3.load(f"./check_points/{RL_model}_{stock_code}")

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

def test_strategy(stock_code, RL_model):

    stock_file = find_file('./data/tushare_data/test', str(stock_code))
    test_env, len_test = prepare_env(stock_file)

    model = load_model(RL_model, stock_code)

    dates, daily_profits, daily_opens, daily_closes, daily_highs, daily_lows = test_model(test_env, len_test, model)
    plot_daily_profits(stock_code, RL_model, daily_profits, dates, daily_opens, daily_closes, daily_highs, daily_lows)

if __name__ == '__main__':

    test_strategy(args['train_args']['stock_code'], args['train_args']['rl_model'])


