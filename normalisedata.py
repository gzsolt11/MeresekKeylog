import pandas as pd
import numpy as np

data = pd.read_csv("D:\Pythongyakorlas\Allamvizsga_neuralnetwork\processed_data\osszesitett6digit.csv")

data = (data-data.min())/(data.max()-data.min())

compression_opts = dict(method='zip',
                        archive_name='out6digit.csv')  
data.to_csv('out.zip', index=False,
          compression=compression_opts)  