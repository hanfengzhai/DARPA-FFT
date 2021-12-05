import time
from scipy import fftpack
import book_format
book_format.set_style()
import kf_book.kf_internal as kf_internal
from kf_book.kf_internal import DogSimulation
from kf_book import book_plots as book_plots
import numpy as np
from matplotlib import pyplot
import scipy.io
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../results')


jj = 9

with open('Lat_new.txt', 'r') as f1:
        data1 = f1.read().split(); floats1 = []
        for elem1 in data1:
            try:
                floats1.append(float(elem1))
            except ValueError:
                pass

lat = np.array(data1, dtype = np.float64);lat = np.array_split(lat, 86)
x1 = lat

with open('Long_new.txt', 'r') as f2:
        data2 = f2.read().split(); floats2 = []
        for elem2 in data2:
            try:
                floats2.append(float(elem2))
            except ValueError:
                pass
            
longdat = np.array(data2, dtype = np.float64);longdat = np.array_split(longdat, 86)
x2 = longdat

x = np.linspace(0, 405, 405)
x_benchmark = np.linspace(0, 405, 405)# 550
xpred = np.linspace(405, 750, 345)#440 - 550
y_lat = x1[jj][0:405]
y_long = x2[jj][0:405]
# y_benchmark = x1[jj][0:550]

y_fft_lat = fftpack.dct(y_lat, norm="ortho")
y_fft_lat[5:] = 0
y_filter_lat = fftpack.idct(y_fft_lat, norm="ortho")

y_fft_long = fftpack.dct(y_long, norm="ortho")
y_fft_long[5:] = 0
y_filter_long = fftpack.idct(y_fft_long, norm="ortho")


t_lat = time.time()

uk_fourier_lat = OrdinaryKriging(
    x, np.zeros(x.shape), y_filter_lat, variogram_model="power"#, exact_values=False
)
y_fft_pred_lat, y_fft_std_lat = uk_fourier_lat.execute("grid", xpred, np.array([0.0]), backend="loop")

time_fourierkriging_lat = time.time() - t_lat


uk_lat = OrdinaryKriging(
    x, np.zeros(x.shape), y_lat, variogram_model="power"#, exact_values=False
)
y_pred_lat, y_std_lat = uk_lat.execute("grid", xpred, np.array([0.0]), backend="loop")

time_kriging_lat = time.time() - t_lat


t_long = time.time()

uk_fourier_long = OrdinaryKriging(
    x, np.zeros(x.shape), y_filter_long, variogram_model="power"#, exact_values=False
)
y_fft_pred_long, y_fft_std_long = uk_fourier_long.execute("grid", xpred, np.array([0.0]), backend="loop")

time_fourierkriging_long = time.time() - t_long


uk_long = OrdinaryKriging(
    x, np.zeros(x.shape), y_long, variogram_model="power"#, exact_values=False
)
y_pred_long, y_std_long = uk_long.execute("grid", xpred, np.array([0.0]), backend="loop")

time_kriging_long = time.time() - t_long


y_pred_lat = np.squeeze(y_pred_lat)
y_std_lat = np.squeeze(y_std_lat)
y_fft_pred_lat = np.squeeze(y_fft_pred_lat)
y_fft_std_lat = np.squeeze(y_fft_std_lat)

y_pred_long = np.squeeze(y_pred_long)
y_std_long = np.squeeze(y_std_long)
y_fft_pred_long = np.squeeze(y_fft_pred_long)
y_fft_std_long = np.squeeze(y_fft_std_long)


dat_24_lat = y_fft_pred_lat[135:161]
dat_26_lat = y_fft_pred_lat[184:207]
dat_28_lat = y_fft_pred_lat[230:253]
dat_30_lat = y_fft_pred_lat[276:299]
dat_2_lat = y_fft_pred_lat[322:345]

dat_24_long = y_fft_pred_long[135:161]
dat_26_long = y_fft_pred_long[184:207]
dat_28_long = y_fft_pred_long[230:253]
dat_30_long = y_fft_pred_long[276:299]
dat_2_long = y_fft_pred_long[322:345]

# =====================================

pred_24_lat = np.mean(dat_24_lat)
pred_26_lat = np.mean(dat_26_lat)
pred_28_lat = np.mean(dat_28_lat)
pred_30_lat = np.mean(dat_30_lat)
pred_2_lat = np.mean(dat_2_lat)

pred_24_long = np.mean(dat_24_long)
pred_26_long = np.mean(dat_26_long)
pred_28_long = np.mean(dat_28_long)
pred_30_long = np.mean(dat_30_long)
pred_2_long = np.mean(dat_2_long)

# ========SAVE FINAL DATA PREDICTION=========

final_pred = [[pred_24_lat, pred_26_lat, pred_28_lat, pred_30_lat, pred_2_lat],[pred_24_long, pred_26_long, pred_28_long, pred_30_long, pred_2_long]]

np.savetxt(('id'+str(jj)+'.txt'),final_pred)
