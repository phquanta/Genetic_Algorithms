# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:37:32 2018

@author: Andrei
"""

import ga
from imp import reload
import matplotlib.pyplot as plt
import numpy as np

reload(ga)



me1=[]
meanAllG0=[]
stddAllG0=[]
meanAllG1=[]
stddAllG1=[]

std1=[]

me2=[]
std2=[]
KuniAll=[]



for i in range(0,30):
    G0 = ga.GA_Abuin(['0','1'],0.005, '2',0.1, 0., 'fps', 0,0, 'ras', 16 )
    G1 = ga.GA_Abuin(['0','1'],0.005, 'u',0.1, 0.8, 'fps', 0,0, 'ras', 16 )

    G0.runMain()
    G1.runMain()



    mean1G0=[np.mean(x) for x in G0.fitAll]
    stddG0=[np.std(x) for x in G0.fitAll]
    mean1G1=[np.mean(x) for x in G1.fitAll]
    stddG1=[np.std(x) for x in G1.fitAll]


    me1.append(mean1G0)
    std1.append(stddG0)
    me2.append(mean1G1)
    std2.append(stddG1)
    KuniAll.extend(G1.Kuni)

meanAllG0=[np.mean([me1[i][j] for i in range(len(me1))]) for j in range(0,100)]
stddAllG0=[np.std([std1[i][j] for i in range(len(std1))]) for j in range(0,100)]
meanAllG1=[np.mean([me2[i][j] for i in range(len(me2))]) for j in range(0,100)]
stddAllG1=[np.std([std2[i][j] for i in range(len(std2))]) for j in range(0,100)]




    
    
f, axarr = plt.subplots(1,2)
axarr[0].errorbar(range(0,100),meanAllG0,stddAllG0, errorevery=3,color='black',ecolor='b',elinewidth=1.5,label='k=2')
axarr[0].errorbar(range(0,100),meanAllG1,stddAllG1, errorevery=3,color='green',ecolor='red',elinewidth=1.5,label='k=Uni')
axarr[0].set_title("Mean of Fitness/Obj. + St.D.")
axarr[1].plot(G0.bestChromosome,label='k=2')
axarr[1].plot(G1.bestChromosome,label='k=Uni')
axarr[1].set_title("Best Fitness/Obj. Val.")
axarr[0].legend(loc='upper right', shadow=True, fontsize='x-large')
axarr[1].legend(loc='upper right', shadow=True, fontsize='x-large')
    
    
plt.savefig('Fig2e_mine_maxone_16_2.pdf')

