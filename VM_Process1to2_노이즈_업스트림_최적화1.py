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
from simulator.VM_Process1_노이즈시뮬레이터 import VM_Process1_노이즈시뮬레이터
from simulator.VM_Process2_시뮬레이터 import VM_Process2_시뮬레이터
from simulator.FDC_Graph import FDC_Graph
import os
# from pandas import DataFrame, Series
# import pandas as pd

os.chdir("D:/10. 대학원/04. Source/09. VM_Source/03. VMOnly/")

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])    #recipe gain matrix
d_p1 = np.array([[0.1, 0], [0.05, 0]])  #drift matrix
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]])) # FDC variable matrix
# Process 변수와 출력 관련 system gain matrix

A_p2 = np.array([[1, 0.1], [-0.5, 0.2]])
d_p2 = np.array([[0, 0.05], [0, 0.05]])
C_p2 = np.transpose(np.array([[0.1, 0, 0, -0.2, 0.1], [0, -0.2, 0, 0.3, 0]]))
F_p2 = np.array([[2, 0], [0, 2]])
SEED1 = 100000  #9, 3000, 4000, 80, 200, 1000, 1000000
SEED2 = 818
# Process 변수와 출력 관련 system gain matrix

M = 10
Z_DoE = 12
Z_VM = 50
Nz_RUN = 15

Upstream_Rule_MAPE = 20

v1_PLS = 0.6
v2_PLS = 0.6


def main():
    fdh_graph = FDC_Graph()
    fwc_p1_vm = VM_Process1_시뮬레이터(A_p1, d_p1, C_p1, SEED1)
    fwc_p1_vm.DoE_Run(lamda_PLS=v1_PLS, Z=Z_DoE, M=M)  # DoE Run
    Normal_VMResult, Normal_ACTResult, ez_run, o_y_act, o_y_prd, _ = fwc_p1_vm.VM_Run(lamda_PLS=v1_PLS, Z=Z_VM, M=M)

    fwc_p1_vm = VM_Process1_노이즈시뮬레이터(A_p1, d_p1, C_p1, SEED1)
    fwc_p1_vm.DoE_Run(lamda_PLS=v1_PLS, Z=Z_DoE, M=M)  # DoE Run
    Error_VMresult, Error_ACTResult, ez_run, p1_y_act, p1_y_prd, _ = fwc_p1_vm.VM_Run(lamda_PLS=v1_PLS, Z=Z_VM, M=M)

    # fdh_graph.plt_show1(Z_VM * M, p1_y_act[:, 0:1], p1_y_prd[:, 0:1])
    # fdh_graph.plt_show2(Z_VM, ez_run[:, 0:1], ez_run[:, 1:2])

    p1_q1_mape_Queue = []

    # metrology 마다 보여주는 MAPE 값이 의미가 없다.
    for z in np.arange(Nz_RUN, Z_VM, 1):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, p1_y_act[((z + 1) * M) - 1][0], p1_y_prd[((z + 1) * M) - 1][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 Every Metrology MAPE After 15 Lot : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))
    p1_q1_mape_Queue = []

    for i in np.arange(Nz_RUN * M, Z_VM * M, 1):
        mape = fdh_graph.mean_absolute_percentage_error(i + 1, p1_y_act[i][0], p1_y_prd[i][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 All MAPE After 15 Lot : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))

    p1_q1_mape_Queue = []

    for z in np.arange(0, Z_VM, 1):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, p1_y_act[((z + 1) * M) - 1][0],
                                                        p1_y_prd[((z + 1) * M) - 1][0])
        p1_q1_mape_Queue.append(mape)

    for z in np.arange(Nz_RUN, Z_VM, 1):
        if p1_q1_mape_Queue[z] >= Upstream_Rule_MAPE:
            x1 = p1_y_act[(z * M) - 1]
            x2 = p1_y_act[((z + 1) * M) - 1]
            s = (x2 - x1) / M
            for k in np.arange(z * M, (z + 1) * M - 1, 1):
                i = k % (z * M) + 1
                p1_y_prd[k] = x1 + s * i


    # np.savetxt("output/p1_mape.csv", p1_q1_mape_Queue, delimiter=",", fmt="%.4f")
    # np.savetxt("output/Error_VMresult1.csv", Error_VMresult, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p1_y_prd.csv", p1_y_prd, delimiter=",", fmt="%.4f")
    # np.savetxt("output/p1_y_act.csv", p1_y_act, delimiter=",", fmt="%.4f")
    # np.savetxt("output/Error_ACTResult.csv", Error_ACTResult, delimiter=",", fmt="%.4f")

    max_mape = 0
    min_mape = 1000000000000
    t = 0
    for m in range(1000):
        SEED2 = t + m
        fwc_p2_act = VM_Process2_시뮬레이터(A_p2, d_p2, C_p2, F_p2, v1_PLS, p1_y_prd, p1_y_act, SEED2)
        fwc_p2_act.DoE_Run(lamda_PLS=v2_PLS, Z=Z_DoE, M=M, f=o_y_act)  # DoE Run ACT값으로 가능
        p2_VM_Output, p2_ACT_Output, p2_ez_run, p2_y_act, p2_y_prd, _ = fwc_p2_act.VM_Run(lamda_PLS=v2_PLS, Z=Z_VM, M=M)

        p2_q2_mape_Queue = []

        for i in np.arange(Nz_RUN * M, Z_VM * M, 1):
            mape = fdh_graph.mean_absolute_percentage_error(i + 1, p2_y_act[i][1], p2_y_prd[i][1])
            p2_q2_mape_Queue.append(mape)

        mape = np.mean(p2_q2_mape_Queue)

        print('SEED2 : ', SEED2, 'Process-2 q2 All MAPE After 15 Lot : {0:.2f}%'.format(mape))

        if mape > max_mape:
            max_mape = mape
            seed_right = SEED2
            print('SEED2 : ', SEED2, ' , max_mape = ', max_mape)
        if mape < min_mape:
            min_mape = mape
            seed_right1 = SEED2
            print('SEED2 : ', SEED2, ' , min_mape = ', min_mape)

    print('seed_right : ', seed_right, ' , max_mape = ', max_mape)
    print('seed_right1 : ', seed_right1, ' , min_mape = ', min_mape)

if __name__ == "__main__":
    main()
