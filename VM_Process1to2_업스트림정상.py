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
from simulator.VM_Process2_시뮬레이터 import VM_Process2_시뮬레이터
from simulator.FDC_Graph import FDC_Graph
# from pandas import DataFrame, Series
# import pandas as pd
import os

os.chdir("D:/10. 대학원/04. Source/09. VM_Source/04. VMOnly/")

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])    #recipe gain matrix
d_p1 = np.array([[0.1, 0], [0.05, 0]])  #drift matrix
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]])) #FDC variable matrix
# Process 변수와 출력 관련 system gain matrix

A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[2, 0], [0, 2]])

# 100000020  651
# 100000000   47
SEED1 = 100000020  #100000020 900000020 100000000 700000000 기존 1개일 때   20000
SEED2 = 5416

M = 10
Z_DoE = 12
Z_VM = 50
v1_PLS = 0.6
v2_PLS = 0.6

def main():
    fdh_graph = FDC_Graph()
    fwc_p1_vm = VM_Process1_시뮬레이터(A_p1, d_p1, C_p1, SEED1)
    fwc_p1_vm.DoE_Run(lamda_PLS=v1_PLS, Z=Z_DoE, M=M)  # DoE Run

    # p1_VM_Output : lambda_pls 반영 10 runs act결과 반영 vm
    # p1_ACT_Output : lambda_pls 반영 10 runs act결과 반영
    # p1_ez_run : prd - act (실제)
    # p1_y_act : lambda_pls 반영 안한 act
    # p1_y_prd : lambda_pls 반영 안하고 10 runs act결과 반영하지 않은 예측 값

    p1_VM_Output, p1_ACT_Output, p1_ez_run, p1_y_act, p1_y_prd, p1_ez_all_run = fwc_p1_vm.VM_Run(lamda_PLS=v1_PLS, Z=Z_VM, M=M)

    p1_q1_mape_Queue = []

    # metrology 마다 보여주는 MAPE 값이 의미가 없다.
    for z in np.arange(0, Z_VM, 1):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, p1_y_act[((z + 1) * M) - 1][0], p1_y_prd[((z + 1) * M) - 1][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 Every Metrology MAPE : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))
    p1_q1_mape_Queue = []

    for i in range(Z_VM * M):
        mape = fdh_graph.mean_absolute_percentage_error(i + 1, p1_y_act[i][0], p1_y_prd[i][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 All MAPE : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))

    fdh_graph.plt_show1(Z_VM * M, p1_y_act[:, 0:1], p1_y_prd[:, 0:1], 'Process-1 Normal Case', '1')
    fdh_graph.plt_show2(Z_VM, p1_ez_run[:, 0:1], p1_ez_run[:, 1:2], 'Process-1 Normal Case')

    # np.savetxt("output/p1_VM_Output.csv", p1_VM_Output, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p1_y_prd.csv", p1_y_prd, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p1_y_act.csv", p1_y_act, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p1_ACT_Output.csv", p1_ACT_Output, delimiter=",", fmt="%.4f")
    np.savetxt("output/p1_ez_all_run_q1.csv", p1_ez_all_run, delimiter=",", fmt="%.4f")

    fwc_p2_act = VM_Process2_시뮬레이터(A_p2, d_p2, C_p2, F_p2, v1_PLS, p1_y_prd, p1_y_act, SEED2)
    fwc_p2_act.DoE_Run(lamda_PLS=v2_PLS, Z=Z_DoE, M=M, f=p1_y_act)  #DoE Run ACT값으로 가능
    p2_VM_Output, p2_ACT_Output, p2_ez_run, p2_y_act, p2_y_prd, p2_ez_all_run = fwc_p2_act.VM_Run(lamda_PLS=v2_PLS, Z=Z_VM, M=M)

    p2_q2_mape_Queue = []
    for z in range(Z_VM):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, p2_y_act[((z + 1) * M) - 1][1], p2_y_prd[((z + 1) * M) - 1][1])
        p2_q2_mape_Queue.append(mape)
    print('Process-2 q2 Every Metrology MAPE : {0:.2f}%'.format(np.mean(p2_q2_mape_Queue)))

    p2_q2_mape_Queue = []

    for i in range(Z_VM * M):
        mape = fdh_graph.mean_absolute_percentage_error(i + 1, p2_y_act[i][1], p2_y_prd[i][1])
        p2_q2_mape_Queue.append(mape)

    print('Process-2 q2 All MAPE : {0:.2f}%'.format(np.mean(p2_q2_mape_Queue)))

    # for i in np.arange(22 * M, 34 * M, 1):
    #     mape = fdh_graph.mean_absolute_percentage_error(i + 1, p2_y_act[i][1], p2_y_prd[i][1])
    #     p2_mape_Queue.append(mape)

    fdh_graph.plt_show1(Z_VM * M, p2_y_act[:, 1:2], p2_y_prd[:, 1:2], 'Process-2 Normal Case', '2')
    fdh_graph.plt_show2(Z_VM, p2_ez_run[:, 0:1], p2_ez_run[:, 1:2], 'Process-2 Normal Case')

    # np.savetxt("output/p2_VM_Output.csv", p1_VM_Output, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p2_y_prd.csv", p1_y_prd, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p2_y_act.csv", p1_y_act, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p2_ACT_Output.csv", p1_ACT_Output, delimiter=",", fmt="%.4f")
    np.savetxt("output/p2_ez_all_run_q2.csv", p2_ez_all_run, delimiter=",", fmt="%.4f")


if __name__ == "__main__":
    main()
