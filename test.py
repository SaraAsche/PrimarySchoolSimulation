import matplotlib.pyplot as plt
import random
import numpy as np

liste = []


for i in range(0,1000):
    k = random.randint(0,10)
    l = random.random()
    liste.append(int(l))

def toCumulative(l):                                                                                               
            n = len(l)                                                                                                      
            dictHist = {}                                                                                                   
            for i in l:                                                                                                     
                if i not in dictHist:                                                                                       
                    dictHist[i] = 1                                                                                         
                else:                                                                                                       
                    dictHist[i] += 1                                                                                        
            cHist = {}                                                                                                      
            cumul = 1                                                                                                       
            for i in dictHist:                                                                                              
                cHist[i] = cumul                                                                                            
                cumul -= float(dictHist[i])/float(n)
            return cHist

power = toCumulative(liste)


plt.plot(power.keys(), power.values())
plt.yscale('linear')
plt.xscale('linear')

plt.show()