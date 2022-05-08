from matplotlib import pyplot as plt


class PlotBuilder:
    def __init__(self, df):
        self.dataframe = df

    def user_rate(self, unit_resulotion):
        arrival_times = self.dataframe['start']
        seconds = (arrival_times / 1000) % 60
        seconds = seconds - min(seconds)
        minutes = (arrival_times / (1000 * 60)) % 60
        minutes = minutes - min(minutes)

        self.dataframe['start_second'] = seconds.astype('int')
        self.dataframe['start_minute'] = minutes.astype('int')
        self.dataframe['start_millis'] = arrival_times - min(arrival_times)
        print(self.dataframe)
