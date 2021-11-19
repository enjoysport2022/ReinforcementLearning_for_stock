import os
import tushare as ts

class Downloader(object):
    def __init__(self,
                 output_dir,
                 codes,
                 train_start,
                 train_end,
                 test_start,
                 test_end):
        self.output_dir = output_dir
        self.codes = codes
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

    def run(self):
        for code in codes:
            df = ts.get_hist_data(code)
            df['date'] = df.index
            df.index = range(len(df))

            train = df.loc[(df['date'] >= train_start) & (df['date'] <= train_end)]
            test = df.loc[(df['date'] >= test_start) & (df['date'] <= test_end)]
            train = train.sort_values(by='date')
            test = test.sort_values(by='date')
            train.index = range(len(train))
            test.index = range(len(test))

            cols = ['date'] + [x for x in test.columns if x != 'date']
            train = train[cols]
            test = test[cols]

            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir + '/train', exist_ok=True)
            os.makedirs(self.output_dir + '/test', exist_ok=True)
            train.to_csv(f'{self.output_dir}/train/{code}.csv',index=False)
            test.to_csv(f'{self.output_dir}/test/{code}.csv',index=False)


if __name__ == '__main__':

    # 获取全部股票的日K线数据
    output_dir = './tushare_data'
    codes = ['600000', '600006']
    train_start = '2019-05-21'
    train_end = '2020-12-31'
    test_start = '2021-01-01'
    test_end = '2021-11-18'

    downloader = Downloader(output_dir,
                            codes,
                            train_start,
                            train_end,
                            test_start,
                            test_end)
    downloader.run()
