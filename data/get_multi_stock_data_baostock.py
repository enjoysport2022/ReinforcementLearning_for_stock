import baostock as bs
import os
import pandas as pd

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok = True)

class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start,
                 date_end):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self, codes):
        stock_df = self.get_codes_by_date(self.date_end)

        if codes is None:
            for index, row in stock_df.iterrows():
                print(f'processing {row["code"]} {row["code_name"]}')
                df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                       start_date=self.date_start,
                                                       end_date=self.date_end,
                                                       adjustflag="1").get_data() # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权
                df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"].replace("*","star_")}.csv', index=False)
        else:
            for index, row in stock_df.iterrows():
                if row["code"] not in codes:
                    continue
                print(f'processing {row["code"]} {row["code_name"]}')
                df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                       start_date=self.date_start,
                                                       end_date=self.date_end,
                                                       adjustflag="1").get_data()  # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权
                df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"].replace("*", "star_")}.csv',
                               index=False)

        self.exit()

if __name__ == '__main__':

    codes = ['sh.600000',
             'sh.600004',
             'sh.600006',
             'sh.600007',
             'sh.600008',
             'sh.600009',
             'sh.600010',
             'sh.600011',
             'sh.600012',
             'sh.600015',
             'sh.600016',
             'sh.600017',
             'sh.600018',
             'sh.600019',
             'sh.600020',
             'sh.600021',
             'sh.600022',
             'sh.600023',
             'sh.600025',
             'sh.600026',
             'sh.600027',
             'sh.600028',
             'sh.600029',
             'sh.600030',
             'sh.600031',
             'sh.600033',
             'sh.600035',
             'sh.600036',
             'sh.600037',
             'sh.600038']

    # 获取全部股票的日K线数据
    datapath = './baostock/multi_stock'
    mkdir(datapath)
    downloader = Downloader(datapath, date_start='2010-01-01', date_end='2020-12-31')
    downloader.run(codes)

    files = os.listdir(datapath)
    for i, file in enumerate(files):
        temp = pd.read_csv(datapath + '/' + file)
        if i == 0:
            res = temp
        else:
            res = res.append(temp)
    res.to_csv(datapath + '/' + 'trading.csv', index = False)

