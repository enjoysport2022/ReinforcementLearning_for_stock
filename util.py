import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from stable_baselines3.common.vec_env import DummyVecEnv
from env.StockTradingEnv import StockTradingEnv
import pandas as pd

def prepare_env(stock_file):
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    return env, len(df)

def find_file(path, name):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)

def plot_daily_profits(stock_code, RL_model, daily_profits, dates, daily_opens, daily_closes, daily_highs, daily_lows):

    fig = make_subplots(rows=2, cols=1, subplot_titles=("profit", "daily price"), shared_xaxes=True)

    fig.add_trace(
        go.Scatter(x=dates, y=daily_profits, mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Candlestick(x=dates,
                       open=daily_opens, high=daily_highs,
                       low=daily_lows, close=daily_closes),
        row=2, col=1
    )
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False, title_text=f"{stock_code}, {RL_model}")
    fig.show()

    # os.makedirs('./img/', exist_ok=True)
    # fig.write_image(f'./img/{stock_code + "_" + RL_model}.png')