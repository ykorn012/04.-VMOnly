#======================================================================================================
# !/usr/bin/env python
# title          : LocalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2019-06-17
# version        : v0.8
# usage          : python LocalW2W_FWC_Run.py
# notes          : Reference Paper "Virtual metrology and feedback control for semiconductor manufacturing"
# python_version : v3.5.3
#======================================================================================================
import numpy as np
from simulator.VM_Process1_시뮬레이터 import VM_Process1_시뮬레이터
from simulator.FDC_Graph import FDC_Graph
# from pandas import DataFrame, Series
# import pandas as pd
import os

os.chdir("D:/10. 대학원/04. Source/09. VM_Source/04. VMOnly/")

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])    #recipe gain matrix
d_p1 = np.array([[0.1, 0], [0.05, 0]])  #drift matrix
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]])) # FDC variable matrix
SEED = 20000

M = 10
Z_DoE = 12
Z_VM = 50
Nz_RUN = 15
v_PLS = 0.6

# Process 변수와 출력 관련 system gain matrix

def main():
    fdh_graph = FDC_Graph()
    fwc_p1_vm = VM_Process1_시뮬레이터(A_p1, d_p1, C_p1, SEED)
    fwc_p1_vm.DoE_Run(lamda_PLS=v_PLS, Z=Z_DoE, M=M)  #DoE Run
    VM_Output, ACT_Output, ez_run, y_act, y_prd, ez_all_run = fwc_p1_vm.VM_Run(lamda_PLS=v_PLS, Z=Z_VM, M=M)

    fdh_graph.plt_show1(Z_VM * M, y_act[:, 0:1], y_prd[:, 0:1], 'Process-1 Normal Case', '1')
    fdh_graph.plt_show2(Z_VM, ez_run[:, 0:1], ez_run[:, 1:2], 'Process-1 Normal Case')

    p1_q1_mape_Queue = []

    #metrology 마다 보여주는 MAPE 값이 의미가 없다.
    for z in np.arange(0, Z_VM, 1):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, y_act[((z + 1) * M) - 1][0], y_prd[((z + 1) * M) - 1][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 Every Metrology MAPE : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))
    p1_q1_mape_Queue = []


    for i in range(Z_VM * M):
        mape = fdh_graph.mean_absolute_percentage_error(i + 1, y_act[i][0], y_prd[i][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 All MAPE : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))
    # np.savetxt("output/Normal_mape_Queue_Output.csv", p1_mape_Queue, delimiter=",", fmt="%.4f")
    # np.savetxt("output/Normal_VM_Output.csv", VM_Output, delimiter=",", fmt="%.4f")
    np.savetxt("output/ez_all_run.csv", ez_all_run, delimiter=",", fmt="%.4f")

if __name__ == "__main__":
    main()
