from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
class PlotBuilder:
    def __init__(self, df):
        self.dataframe = df
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

    def average_latency_second(self):
        render_data = self.dataframe.groupby('start_time')['latency'].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        axb = ax.twinx()
        axb.set_ylabel('latency')
        axb.plot([i for i in range(1,len(render_data)+ 1)], render_data, color='black', label='pressure')
        plt.savefig('b.png')
    def user_rate_second(self):
        render_data = self.dataframe.groupby(['start_time']).size()
        fig, ax = plt.subplots(figsize=(10, 6))
        axb = ax.twinx()
        axb.set_ylabel('users')
        axb.plot([i for i in range(1,len(render_data)+ 1)], render_data, color='blue', label='rate')
        plt.savefig('a.png')       

    def user_rate_milliseconds(self, interval):
        temp_df = self.dataframe[self.dataframe['start_time']==self.dataframe.iloc[0]['start_time']]
        render_data = temp_df.groupby(pd.cut(temp_df["start_millis"], np.arange(0, max(temp_df['start_millis'])+interval, interval))).size()
        plt.plot([i for i in range(len(render_data))], render_data)
        plt.savefig('c.png')
    
    def user_rate_second_interval(self, interval):
        temp_df = self.dataframe
        render_data = temp_df.groupby(pd.cut(temp_df["start_time"], np.arange(0, max(temp_df['start_time'])+interval, interval))).mean()
        return render_data

    def user_rate(self, unit_resulotion):
        arrival_times = self.dataframe['start']
        seconds = (arrival_times / 1000) % 60
        seconds = seconds - min(seconds)
        minutes = (arrival_times / (1000 * 60)) % 60
        minutes = minutes - min(minutes)

        self.dataframe['start_second'] = seconds.astype('int')
        self.dataframe['start_minute'] = minutes.astype('int')
        self.dataframe['start_time'] = self.dataframe['start_second'] + 60*self.dataframe['start_minute'] 
        self.dataframe['start_millis'] = arrival_times - min(arrival_times)
        if unit_resulotion == 's':
            pass

df = pd.read_csv('users.csv')
pb = PlotBuilder(df)
pb.user_rate('s')
pb.user_rate_milliseconds(1)