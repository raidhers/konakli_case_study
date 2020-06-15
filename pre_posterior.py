import numpy as np
import math as m
from scipy import random
from scipy.stats import norm
import matplotlib.pyplot as plt
#DEFININIG VARIBLES AND CONSTANTS

Mu_R = 2.5 #Mean of lognormal resistance
CoV_R = 0.1 #Coeefficent of variations of log normal resistance
Stdv_R = Mu_R*CoV_R #standard deviation of lognormal resistance

#Normal Parameters for resistance
Mean_LNR = 2 * m.log(Mu_R) - 1/2 * m.log((Stdv_R)**2 + Mu_R**2)
Sd_LNR = m.sqrt(-2*m.log(Mu_R)+m.log(Stdv_R**2 + Mu_R**2))

Mu_S = 1.0 #Mean of lognormal load
CoV_S = 0.3 #Coeefficent of variations of log normal load
Stdv_S = Mu_S * CoV_S #standard deviation of lognormal load

#Normal Parameters for load
Mean_LNS = 2*m.log(Mu_S)-1/2*m.log((Stdv_S)**2 + Mu_S**2)
Sd_LNS = m.sqrt(-2*m.log(Mu_S)+m.log(Stdv_S**2 + Mu_S**2))
Sd_LNS2 = m.sqrt(Sd_LNS)
Sd_LNRV = np.log(1+0.1**2)

#Parametrs for structral reliabitly (G = R-S)
Sd_LNG = m.sqrt(Sd_LNR**2 + Sd_LNS**2)

#Parameters of error sampling
Mean_Er = 0
Sd_Er = 0.5*Sd_LNR

#Constants
C_F = 1000 #Cost of Failure (1000 units)
C_R = 10   #Replacement Cost (10 units)
LAMBDA_VAL = 0.02 #Annual interest rate
K = 0.01 #Deterministic Parameters
v = 1    #Deterministic Parameters
def a_val(): #Degradation of component with respect to time
  avalues = []     #Initializing dictionary to hold values from a0 to a30 (a -> degradation of the component)
                    #Time t from 0 to 30 years and roudning off to 8 decimal places

  for t in range(31):
      a = np.log((1 - (K*(t**v))))
      avalues.append(a)
  return np.array(avalues)

def m_val_prior(a): #Finding the mean of G in order to find the probabilty of failure
    mean = (Mean_LNR + a) - Mean_LNS
    return mean

def m_val(a, vecVal): #Finding the mean of G in order to find the probabilty of failure
    mean = (np.log(vecVal) + a) - Mean_LNS
    return mean

def m_val_part2(a, LNR): #Finding the mean of G in order to find the probabilty of failure
    mean = (LNR + a) - Mean_LNS
    return mean

def p_of_f(mean_vals): #Finding probability of failure using norm.cdf ()

    pf_val = norm.cdf(0,mean_vals,Sd_LNG)
    return pf_val

def p_of_f_post (mean_vals, Zv):
    pf_val = norm.cdf(0, mean_vals, np.sqrt(Zv))
    return pf_val

def cost_aaa(pfval): #Finding the cost of decsion a0 for every year of decsion
    cost_a0 =[]
    for d in range (30):
        sum = 0
        for i in range (d, 30):
            p0f = pfval[i+1] - pfval[i]
            lam = (1 + LAMBDA_VAL)**(i-d)
            sum = sum + p0f/lam
        cost_a0.append(sum*C_F)
    return cost_a0

def cost_aaa1(pfval):#Finding the cost of decsion a1 for every year of decsion
    cost_a1 =[]
    for d in range (30):
        sum = 0
        for i in range (d, 30):
            p0f = pfval[i+1-d] - pfval[i-d]
            lam = (1 + LAMBDA_VAL)**(i-d)
            sum = sum + p0f/lam
        cost_a1.append(C_R +sum*C_F)
    return cost_a1

def min_of_year(cost_a0,cost_a1): #Taking the minimum costs between a0 and a1

  min_of_c1_c2 = np.minimum(cost_a0,cost_a1)
  return min_of_c1_c2

def C_prior_calc():
    a = a_val()
    m = m_val_prior(a)
    pofF = p_of_f(m)
    Ca_0 = cost_aaa(pofF)
    Ca_1 = cost_aaa1(pofF)
    minV = np.array(min_of_year(Ca_0, Ca_1))
    return minV


def part2 ():
    c_of = np.arange(0.1, 10.1, 0.1)
    measurements = [1,2,5]
    fin_EVI_20 = []
    fin_EVI = []
    for measure in measurements:
        EVI = []
        EVI_20_final = []
        N = measure
        for c in c_of:
            std = c*np.sqrt(Sd_LNRV)
            err_vec = np.random.normal(0,std,N)
            calc = start_func_2(N, err_vec, std)
            mean = calc[0]
            fin_20 = calc[1]
            C_prior = C_prior_calc()
            EVI_20_final.append((C_prior-np.mean(fin_20))/2.2)
            temp = (C_prior-mean)/2.2
            EVI.append(temp[20])
        fin_EVI_20.append(EVI_20_final)
        fin_EVI.append(EVI)
        return fin_EVI_20
        print(fin_EVI_20)

def start_func_2(N,err,std):
    final_array = []
    final_array_20 = []
    samples = np.random.lognormal(Mean_LNR,Sd_LNR,int(1e4))
    for index, r in enumerate(samples):
        sam_xm = np.log(r)*np.ones(N) + err
        LNRS = (np.sqrt(Sd_LNRV) ** -2 + N / std ** 2) ** -0.5
        LNR = (Mean_LNR / Sd_LNRV + np.sum(sam_xm) / std ** 2) * LNRS ** 2
        LNRV = LNRS**2
        mean2_value = m_val_part2(a_val(), LNR)
        mean_prior = m_val_prior(a_val())
        p_off = p_of_f_post(mean2_value, (LNRV+np.log(1+0.3**2)))
        p_prior = p_of_f(mean_prior)
        a0 = cost_aaa(p_off)
        a1 = cost_aaa1(p_prior)
        mini = min_of_year(a0,a1)
        final_array.append(mini)
        final_array_20.append(mini[20])
    mean = np.average(final_array,axis=0)
    return [mean, final_array_20]

#plt.figure(4)
#plt.plot(c_of,fin_EVI[1], label='$EVI$ with n=2')
#plt.plot(c_of,fin_EVI[2], label='$EVI$ with n=5')
#plt.xscale('log')
#plt.title('EVI')
#plt.xlabel('Time [years]')
#plt.ylabel('$EVI$ [u]')
#plt.xlim([0.1,10.1])
#plt.grid()
#plt.legend()
#plt.show()
