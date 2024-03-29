import numpy as np
from matplotlib import pyplot as plt

class FDC_Graph:

    def plt_show1(self, n, y_act, y_prd, subtitle, q_param):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y_act, 'rx--', label='$y_{act}^{' + q_param + '}$', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y_prd, 'bx--', label='$y_{pred}^{' + q_param + '}$', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.xlabel('Run No.')
        plt.ylabel('Actual and Predicted Response')
        plt.show()

    def plt_show2(self, n, y1, y2, subtitle, color1='bx-', color2='gx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(0, n + 1, 1), y1, color1, label='$y^{1}$', lw=2, ms=5, mew=2)
        plt.plot(np.arange(0, n + 1, 1), y2, color2, label='$y^{2}$', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-1.2, 1.3, 0.2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')

    def plt_show3(self, n, y1, y2):
        plt.figure()
        plt.plot(np.arange(n), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-12, 3, 2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')

    def plt_show4(self, n, y1):
        plt.figure()
        plt.plot(np.arange(n), y1, 'rx-', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-1.2, 1.3, 0.2))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')

    def mean_absolute_percentage_error(self, z, y_act, y_prd):
        #print('z: ', z, 'y_act : ', y_act, 'y_prd : ', y_prd)
        mape = np.mean(np.abs((y_act - y_prd) / y_act)) * 100
        #print('mape : ', mape)
        return mape

    def plt_show12(self, n, y1, y2):
        plt.figure()
        plt.plot(np.arange(0, n + 1, 1), y1, 'bx-', y2, 'gx--', lw=2, ms=5, mew=2)
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-8, 6.1, 1))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')

    def plt_show2_1(self, n, y1, y2, subtitle, color1='bx-', color2='gx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(0, n + 1, 1), y1, color1, label='Normal ($y^{1}$)', lw=2, ms=5, mew=2)
        plt.plot(np.arange(0, n + 1, 1), y2, color2, label='Abnormal ($y^{1}$)', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 5))
        plt.yticks(np.arange(-8, 6.1, 1))
        plt.xlabel('Metrology Run No.(z)')
        plt.ylabel(r'Prediction Error $(y_{z} - \hat y_{z})$')

    def plt_show2_2(self, n, y1, y2, subtitle, q_param, color1='bx-', color2='gx--'):
        plt.figure()
        plt.suptitle(subtitle, fontsize=15)
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.91)
        plt.plot(np.arange(n), y1, color1, label='Normal ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.plot(np.arange(n), y2, color2, label='Abnormal ($y^{' + q_param + '}$)', lw=2, ms=5, mew=2)
        plt.legend(loc='upper left', fontsize='large')
        plt.xticks(np.arange(0, n + 1, 50))
        plt.yticks(np.arange(-23, 15, 5))
        plt.xlabel('Run No.')
        plt.ylabel(r'Prediction Error $(y - \hat y)$')
