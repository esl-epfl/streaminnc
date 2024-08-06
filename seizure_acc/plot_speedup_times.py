import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme()
cm = 1/2.54

figsize = (3 * cm, 3 * cm)

fontsize = 6

markersize = 2.5
linewidth = 1.5

plt.rc('font', size = fontsize)          # controls default text sizes
plt.rc('axes', titlesize = fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize = fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize = fontsize)    # fontsize of the tick labels
plt.rc('legend', fontsize = fontsize)    # legend fontsize
plt.rc('figure', titlesize = fontsize)  # fontsize of the figure title

plt.rcParams.update({"font.family" : "Times New Roman"})

save_figures = False

ratios_buffer = [1/2, 1/3, 1/4, 1/5, 1/6, 
          1/8, 1/10, 1/12, 1/15, 
          1/16, 1/20]
ratios_buffer = np.array(ratios_buffer)

initial_time = 132007
initial_input_size = 960

exec_time_buffer = [73_371, 49_091, 36_805, 29_307, 24_541,
             18_381, 14_771, 12_602, 9_962, 9_246, 
             7_747]
exec_time_buffer = np.array(exec_time_buffer)

ratios = [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/10, 1/12]
ratios = np.array(ratios)

exec_time = [66_527, 44_559, 33_455, 26_804, 22_222, 
             16_752, 13_422, 11_295]
exec_time = np.array(exec_time)


plt.figure(figsize = figsize)
plt.plot(ratios * initial_input_size, exec_time / 1000, '-o',
         linewidth = linewidth, markersize = markersize)
plt.plot(ratios_buffer * initial_input_size, 
         exec_time_buffer / 1000, '-o',
         linewidth = linewidth, markersize = markersize)
plt.plot(ratios * initial_input_size, initial_time * ratios / 1000, 
         '--', linewidth = linewidth, markersize = markersize)
plt.ylabel('Execution Time (ms)')
plt.xlabel('Input Size (time points)')

if save_figures:
    plt.savefig('./results/figures/speedup_times.svg', bbox_inches = 'tight')
    
    
x = ratios * initial_input_size
y = exec_time / 1000
reg = LinearRegression().fit(x[:, None], y[:, None])

print(reg.coef_)

x = ratios_buffer * initial_input_size
y = exec_time_buffer / 1000
reg = LinearRegression().fit(x[:, None], y[:, None])

print(reg.coef_)

x = ratios * initial_input_size
y = initial_time * ratios / 1000
reg = LinearRegression().fit(x[:, None], y[:, None])

print(reg.coef_)