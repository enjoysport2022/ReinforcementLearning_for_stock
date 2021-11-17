import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def find_file(path, name):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False

def plot_daily_profits(stock_code, daily_profits):
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    os.makedirs('./img/', exist_ok=True)
    plt.savefig(f'./img/{stock_code}.png')
    plt.show()