''' plotter

Script for the generation of the plots in Fig. 4 of the paper.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rc('font', family='serif')

### IMPORTING THE DATASETS (XXZ model)
loss100 = pd.read_csv('./output/paper/XXZ_100_BKT/test/loss_anomaly_detection.csv')
loss160 = pd.read_csv('./output/paper/XXZ_160_BKT/test/loss_anomaly_detection.csv')
loss200 = pd.read_csv('./output/paper/XXZ_200_BKT/test/loss_anomaly_detection.csv')
loss400 = pd.read_csv('./output/paper/XXZ_400_BKT/test/loss_anomaly_detection.csv')

from matplotlib import cm

plt.figure(1,figsize=(8,6))
plt.xlim([-1.7,0])
plt.ylim([0,35])
plt.ylabel('$\mathcal{L}_{rec}-$Noise',fontsize=22)
plt.xlabel('$\Delta/J$',fontsize=22)
plt.xticks(fontsize=21)
plt.yticks(np.arange(0, 40, step=10),fontsize=21)
plt.plot(loss100['parameter'],loss100['loss-noise'],'o-',ms=4,label="$L=100$",color=plt.cm.tab20c(0))
plt.plot(loss160['parameter'],loss160['loss-noise'],'o-',ms=4,label="$L=160$",color=plt.cm.tab20c(4))
plt.plot(loss200['parameter'],loss200['loss-noise'],'o-',ms=4,label="$L=200$",color=plt.cm.tab20c(8))
plt.plot(loss400['parameter'],loss400['loss-noise'],'o-',ms=4,label="$L=400$",color=plt.cm.tab20c(12))
plt.legend(fontsize=16)
plt.grid()
plt.axvline(-1,lw=5,ls='--',color='red')
plt.axvspan(-0.65, 0, facecolor='yellow', alpha=0.15)
plt.axvspan(-0.8, -0.65, facecolor='green', alpha=0.15)
plt.text(-0.325,13,'\\textbf{TRAINING SET}',ha='center',va='center',fontsize='18',color='#FF8000',rotation=90)
plt.text(-0.72,13,'\\textbf{VALIDATION SET}',ha='center',va='center',fontsize='18',color='#007000',rotation=90)
plt.text(-1.05,18,'\\textbf{TRANSITION}',ha='center',va='center',fontsize='18',color='red',rotation=90)
plt.gcf().text(0.05, 0.85, '$(a)$', fontsize=22)

### IMPORTING THE DATASETS (BH model)
loss32 = pd.read_csv('./output/paper/BH_32_BKT/test/loss_anomaly_detection.csv')
loss64 = pd.read_csv('./output/paper/BH_64_BKT/test/loss_anomaly_detection.csv')
loss100 = pd.read_csv('./output/paper/BH_100_BKT/test/loss_anomaly_detection.csv')
loss192 = pd.read_csv('./output/paper/BH_192_BKT/test/loss_anomaly_detection.csv')
loss256 = pd.read_csv('./output/paper/BH_256_BKT/test/loss_anomaly_detection.csv')

plt.figure(2,figsize=(8,6))
plt.xticks(fontsize=21)
plt.yticks(np.arange(0, 40, step=10),fontsize=21)
plt.xlim([0,5])
plt.ylim([0,35])
plt.ylabel('$\mathcal{L}_{rec}-$Noise',fontsize=18)
plt.xlabel('$U/J$',fontsize=22)
plt.plot(loss32['parameter'],loss32['loss-noise'],'o-',ms=4,label="$L=32$",color='#E57439')
plt.plot(loss64['parameter'],loss64['loss-noise'],'o-',ms=4,label="$L=64$",color='#EDB732')
plt.plot(loss100['parameter'],loss100['loss-noise'],'o-',ms=4,label="$L=100$",color='#229487')
plt.plot(loss192['parameter'],loss192['loss-noise'],'o-',ms=4,label="$L=192$",color='#5387DD')
plt.plot(loss256['parameter'],loss256['loss-noise'],'o-',ms=4,label="$L=256$",color='#A12864')
plt.grid()
plt.legend(fontsize=16)
plt.axvspan(0, 2.5, facecolor='yellow', alpha=0.15)
plt.axvspan(2.5, 3, facecolor='green', alpha=0.15)
plt.axvline(3.39,lw=5,ls='--',color='red')
plt.text(3.55,28,'\\textbf{TRANSITION}',ha='center',va='center',fontsize='18',color='red',rotation=90)
plt.text(1.2,12,'\\textbf{TRAINING SET}',ha='center',va='center',fontsize='18',color='#FF8000',rotation=90)
plt.text(2.75,17,'\\textbf{VALIDATION SET}',ha='center',va='center',fontsize='18',color='#007000',rotation=90)
plt.gcf().text(0.05, 0.85, '$(b)$', fontsize=22)

### IMPORTING THE DATASETS (BH2S model)
loss32 = pd.read_csv('./output/paper/BH2S_32_BKT/test/loss_anomaly_detection.csv')
loss64 = pd.read_csv('./output/paper/BH2S_64_BKT/test/loss_anomaly_detection.csv')
loss96 = pd.read_csv('./output/paper/BH2S_96_BKT/test/loss_anomaly_detection.csv')
loss128 = pd.read_csv('./output/paper/BH2S_128_BKT/test/loss_anomaly_detection.csv')

U = 10
plt.figure(3,figsize=(8,6))
plt.xlim([-0.4,0])
plt.ylim([0,35])
plt.xlabel('$U_{AB}/U$',fontsize=22)
plt.ylabel('$\mathcal{L}_{rec}-$Noise',fontsize=22)
plt.xticks(fontsize=21)
plt.yticks(np.arange(0, 35, step=10),fontsize=21)
plt.plot(loss32['parameter']/U,loss32['loss-noise'],'o-',ms=4,label="$L=32$",color=plt.cm.tab20c(0))
plt.plot(loss64['parameter']/U,loss64['loss-noise'],'o-',ms=4,label="$L=64$",color=plt.cm.tab20c(4))
plt.plot(loss96['parameter']/U,loss96['loss-noise'],'o-',ms=4,label="$L=100$",color=plt.cm.tab20c(8))
plt.plot(loss128['parameter']/U,loss128['loss-noise'],'o-',ms=4,label="$L=192$",color=plt.cm.tab20c(12))
plt.grid()
plt.legend(fontsize=16)
plt.axvspan(-0.1, 0, facecolor='yellow', alpha=0.15)
plt.axvspan(-0.15, -0.1, facecolor='green', alpha=0.15)
plt.gcf().text(0.05, 0.85, '$(c)$', fontsize=22)
plt.text(-0.05,18,'\\textbf{TRAINING SET}',ha='center',va='center',fontsize='18',color='#FF8000',rotation=90)
plt.text(-0.125,18,'\\textbf{VALIDATION SET}',ha='center',va='center',fontsize='18',color='#007000',rotation=90)
plt.show() 
# %%
