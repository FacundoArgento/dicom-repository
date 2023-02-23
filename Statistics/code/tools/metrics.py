import numpy as np
from . import __init__
from statistics import pstdev
from data.store import save_csv_file


class Metronome():

    def __init__(self):
        # Create a dict for store all metrics
        self.__d = {}

    def add_metric(self, name, value):
        ''' Add mertic for track. '''

        if name not in self.__d.keys():
            self.__d[name] = []
        try:
            # Value could be a list of numbers
            self.__d[name].extend(value)
        except:
            # or a single number
            self.__d[name].append(value)

    def get_values(self, name):
        ''' Get all values for a certain metric name or None if this doesn\'t exist. '''
        
        return self.__d[name] if name in self.__d.keys() else None

    def metric_names(self):
        ''' Return a list with the names of metric tracked. '''

        return [k for k in self.__d.keys()]

    def save_statistics(self, path, filename='metrics', start_index=1):
        ''' Save the mean (std) of all metric tracked in the specified path. '''

        p = {}
        for name in self.metric_names()[start_index:]:
            mean = np.mean((self.get_values(name)))
            mean = round(mean, 2) if 'lv co' in name.lower() else int(mean)
            std = round(pstdev(self.get_values(name)), 2)
            
            p[name] = f'{mean} ({std})'
            
        save_csv_file([p], path, filename)

        return path
    
    def save_metrics(self, path, filename='metrics'):
        ''' Save metrics. The metrics to save should be the same size of the 'Caso' values. '''
        
        rows = []
        count_rows = len(self.get_values('Caso'))

        for r in range(count_rows):
            row = {}
            for name in self.metric_names():
                row[name] = self.get_values(name)[r]
            rows.append(row)
            
        save_csv_file(rows, path, filename)

        return path
