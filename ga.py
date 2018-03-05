# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:11:32 2018

@author: Andrei
"""
import random as rnd
import numpy as np
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin

from datetime import datetime
class GA_Abuin:
    rnd.seed(datetime.now())
    
    def __init__(self, alphabet, pMut=0.1, c='1',pCross=0.1, pUCross = 0, sel='fps', tsSize=2, elC=0, prob='ras', length=16 ):
         self.popSize,self.maxGen = 100, 100
         self.tsSize = tsSize
         self.Kuni=[]       # to keep count of K values for uniform crossover
         self.eps = 0
         self.ind=[]
         self.new_pop=[]
         self.old_pop=[]
         self.fitAll=[]
         self.bestChromosome=[] # best Fitness value actually
         self.bestChromosomeBin=[] # best Chromosome in binary form
         self.alphabet = alphabet 
         self.pMut= pMut
         self.seFun  = sel
         self.pCross = pCross 
         self.problem = prob
         self.dim = length//16 # determine dimensions of the problem
         try:
           self.kP =int(c)  
           self.pUCross = 0
         except ValueError:
           self.pUCross = pUCross
           self.kP=0


         self.BigConst = 10.*self.dim+sum([26 for x in range(0, self.dim)])
         


         self.eC = elC
         self.length=length
         self.pop = self.reinit()
    
         #self.pop = [['1' 
         #                    for x in range(self.length)]  
         #                       for y in range(self.popSize)
         #                           ]   
         
    
      #   if sel == 'fps' :   
         self.fitnessFun = getattr(self, 'sel_'+sel+'_'+prob)
        
         
                    
    def reinit(self):
        return [[self.alphabet[rnd.randint(0, len(self.alphabet)-1)] 
                             for x in range(self.length)]  
                                for y in range(self.popSize)
                                    ]                                  
         
         
    def ts(self):
        print('I\'m in ts')
    
    def maxone(self):
        return 


    def  fitness_ras(self,pop):
        xdim=self.length//self.dim
        fVals=[]
        fObj=[]
        for i in range(len(pop)):
            #xGC=[''.join(pop[i])[j*xdim:(j+1)*xdim] for j in range(0,self.dim)]
            #print(xdim)
            self.xGC=[gray_to_bin(''.join(pop[i]))[(j*xdim):((j+1)*xdim)] for j in range(0,self.dim)]
            #print(pop)
            #self.xGC=[(''.join(pop[i]))[(j*xdim):((j+1)*xdim)] for j in range(0,self.dim)]
            
            #print(self.xGC)
            x=[int(i,2)/8192.-4. for i in  self.xGC]
            rhs = [m**2.-10.*np.cos(2.*np.pi*m) for m in x]
            f_x = 10.*self.dim+sum(rhs)
            fVals.append(f_x)
            fObj.append(self.BigConst-f_x)
            
            #print(xGC)
            #print(x)
            #print(f_x)
            #wait = input("PRESS ENTER TO CONTINUE.")
            
        return fVals[:],  fObj[:]
        
        
        
        
    def sel_fps_maxone(self, pop):
        fitnessVals = [len(list(filter(lambda x: x=='1', pop[x]))) for x in range(0,self.popSize)]
        self.totalFitness = sum(fitnessVals)
        self.fitProbs = [float(x) /float(self.totalFitness) for x in fitnessVals]
        self.ind = np.random.choice([x for x in range(0,self.popSize)],self.popSize, replace=True, p=self.fitProbs)
        popNew = [pop[i] for i in self.ind]
        self.fitnessVals = [len(list(filter(lambda x: x=='1', popNew[x]))) for x in range(0,self.popSize)]
        return popNew        
                


                
    

    def sel_fps_ras(self, pop):
        fitnessVals,  ObjVals=self.fitness_ras(pop)

        self.totalFitness = sum(ObjVals)
        self.fitProbs = [float(x)/float(self.totalFitness) for x in ObjVals]
        
        
#        print('haha')
   

        self.ind = np.random.choice([x for x in range(0,len(pop))],len(pop), replace=True, p=self.fitProbs)
        
        
        popNew = [pop[i] for i in self.ind]

        self.fitnessVals,  self.ObjVals=self.fitness_ras(popNew)
        return popNew
   


    def sel_ts_maxone(self, pop):
        popNew=[]
        for i in range(0,self.popSize):
            memSel=[rnd.randint(0,self.popSize-1) for x in range(0,self.tsSize)]
            fitVals=[len(list(filter(lambda x: x=='1', pop[x]))) for x in memSel]
            maxFit= max(fitVals)
            ind = memSel[np.where(np.asarray(fitVals)==maxFit)[0][0]]
            popNew.append(pop[ind])
#            print(fitVals,maxFit, memSel,self.tsSize)
        
        self.fitnessVals = [len(list(filter(lambda x: x=='1', popNew[x]))) for x in range(0,self.popSize)]


        #wait = input("PRESS ENTER TO CONTINUE.")
        return popNew


    def sel_ts_ras(self, pop):
        popNew=[]
        
        for i in range(0,self.popSize):
            memSel=[rnd.randint(0,self.popSize-1) for x in range(0,self.tsSize)]
            memb=[pop[x] for x in memSel]
            fitVals,  ObjVals=self.fitness_ras(memb)
            maxFit= min(fitVals)
            ind = memSel[np.where(np.asarray(fitVals)==maxFit)[0][0]]
            popNew.append(pop[ind])
#            print(fitVals,maxFit, memSel,self.tsSize)
        self.fitnessVals,  self.ObjVals=self.fitness_ras(popNew)


        #wait = input("PRESS ENTER TO CONTINUE.")
        return popNew



    def mutate(self, chromosome):
        newChromosome=[]
        for x in chromosome:
            if rnd.random() < self.pMut and x not in '.':
                if len(self.alphabet) > 2:
                        diffSet =   list(set(self.alphabet).difference(x))[0]  
                        x = diffSet[rnd.randint(0, len(diffSet)-1)]
                else:
                        x=list(set(self.alphabet).difference(x))[0]
            newChromosome.append(x)           
        return newChromosome
                        
    
    
    def crossover(self, p1, p2):
      off1=[]
      off2=[]
      crossA = []     
      if rnd.random()< self.pCross:

         if self.kP > 0:
             kPs = list({rnd.randint(0,self.length) for x in range(self.kP)})
             crossA = kPs[:] 
            
         elif self.kP == 0: 
             kPs=[]
             for x in p1:
                if rnd.random() < self.pUCross:
                 kPs.append(rnd.randint(0,self.length))
                 
             crossA = list(set(kPs))
             self.Kuni.append(len(crossA))
#             print("kPS")
#             print(kPs)
         
         #if self.kP >0: print(crossA,len(crossA),self.Kuni, self.kP)        
         crossA.sort()
         
         if(crossA and crossA[0] ==0): 
             p1,p2 = p2,p1
         else:
              crossA.insert(0,0)
         if crossA[-1] != self.length:   crossA.insert(len(kPs), self.length)
            
      
         crossA.sort()
      
      
      
      #self.Kuni=self.Kuni.append(len(crossA))         
      for i in range(len(crossA)-1):
            indx1=crossA[i]
            indx2=crossA[i+1]
            if i % 2 ==0:
                    off1.extend(p1[indx1:indx2])
                    off2.extend(p2[indx1:indx2])
            else:
                    off1.extend(p2[indx1:indx2])
                    off2.extend(p1[indx1:indx2])
      if not off1: 
          off1=p1[:]
          off2=p2[:]

     
      return off1, off2   
            
    
    def elPop(self,pop,fVals):

            elpop = []
            temp1=[]
            temp2=[]


            fitVals = fVals[:]
            if self.problem == 'maxone' :   fitVals.sort(reverse=True)
            if self.problem == 'ras' :   fitVals.sort(reverse=False)
            elChoose=fitVals[:self.eC]




            for el in elChoose:
                temp1.extend([fitVals[np.where(np.asarray(fVals)==el)[0][j]] for j in range(len(np.where(np.asarray(fVals)==el)[0]))])
                temp2.extend([np.where(np.asarray(fVals)==el)[0][j] for j in range(len(np.where(np.asarray(fVals)==el)[0]))])
                        
            ind = temp2[0:self.eC]
                          
            elpop=[pop[i] for i in ind]
            
            
            #print elpop
            #print fitVals
            #wait = input("PRESS ENTER TO CONTINUE.")
            
            return elpop


    def runMain(self):
        self.old_pop=self.pop[:]
        self.new_pop=self.fitnessFun(self.old_pop)

        if self.problem == 'maxone':  
            self.bestChromosome.append(max(self.fitnessVals))
            mem = self.new_pop[np.where(np.asarray(self.fitnessVals)==max(self.fitnessVals))[0][0]]
            self.bestChromosomeBin.append(''.join(mem))
        
        if self.problem == 'ras':  
            self.bestChromosome.append(min(self.fitnessVals))
            mem = self.new_pop[np.where(np.asarray(self.fitnessVals)==max(self.fitnessVals))[0][0]]
            self.bestChromosomeBin.append(gray_to_bin(''.join(mem)))
        
            
            
        if self.eC > 0:   
             ind=rnd.randint(0,self.popSize)
             self.new_pop[ind:ind+self.eC]=self.elPop(self.old_pop,self.fitnessVals)
            
        cnt=0
        while True:
#            self.new_pop=self.fitnessFun(self.old_pop)
 #           self.bestChromosome.append(max(self.fitnessVals))
            cnt=cnt+1
#            print(cnt)
            self.fitAll.append(self.fitnessVals)
            if self.problem == 'maxone' or self.problem == 'ras':
                 if cnt == self.maxGen: #or max(self.fitnessVals)==self.length:
                     break
 
            crosMates =  [(i,i+1) for i in range(0,self.popSize,2)]
            for i in range(len(crosMates)):
                 m = crosMates[i][0]
                 p = crosMates[i][1]
                 self.new_pop[m], self.new_pop[p] = self.crossover(self.new_pop[m],self.new_pop[p])                
                
            for i in range(0,self.popSize):
                self.new_pop[i]=self.mutate(self.new_pop[i])


            self.old_pop= self.new_pop[:]      
            self.new_pop=self.fitnessFun(self.new_pop)
            
            #if self.problem == 'maxone':  
            #    self.fitnessVals = [len(list(filter(lambda x: x=='1', self.new_pop[x]))) for x in range(0,self.popSize)]
  #          if self.problem == 'ras':                  

                
            if self.eC > 0:   
                ind=rnd.randint(0,self.popSize)
                self.new_pop[ind:ind+self.eC]=self.elPop(self.new_pop,self.fitnessVals)
            if self.problem == 'maxone':  
                self.bestChromosome.append(max(self.fitnessVals))
                mem = self.new_pop[np.where(np.asarray(self.fitnessVals)==max(self.fitnessVals))[0][0]]
                self.bestChromosomeBin.append(''.join(mem))
            if self.problem == 'ras':  
                self.bestChromosome.append(min(self.fitnessVals))
                mem = self.new_pop[np.where(np.asarray(self.fitnessVals)==min(self.fitnessVals))[0][0]]
                #self.bestChromosomeBin.append(''.join(mem))
                self.bestChromosomeBin.append(gray_to_bin(''.join(mem)))


             
             
     
                 
                 
                 
             
            
        
        
        
        
        
        
        
        
        
        
    
    