import prior_analysis as prior
import pre_posterior as pre
import numpy as np
import matplotlib.pyplot as plt
from scipy import random
from scipy.stats import norm


a = prior.a_val()
m = prior.m_val_prior(a)
pofF = prior.p_of_f(m)
Ca_0 = prior.cost_aaa(pofF)
p0f = prior.p0_f(pofF)
Ca_1 = prior.cost_aaa1(pofF)
minV = np.array(prior.min_of_year(Ca_0,Ca_1))
start_func = prior.start_func()
EVI = minV -start_func
#x = pre.part2()


x_val = []
for i in range(30):
  x_val.append(i)

plt.figure(1)
plt.plot(x_val,p0f)
plt.xlabel('Time [years]')
plt.ylabel('$ P_f^0(T_i) $')
plt.ylim([0,0.005])
plt.grid()
plt.show()

plt.figure(2)
plt.plot(x_val,Ca_0, label='$C(a_0)$')
plt.plot(x_val,Ca_1, label='$C(a_1)$')
plt.legend()
plt.title('Cost functions')
plt.xlabel('d')
plt.ylabel('$C(a_i)$')
plt.ylim([0,35])
plt.grid()
plt.show()

plt.figure(3)
plt.plot(x_val, EVI,label='$EVI$')
plt.title('EVI')
plt.xlabel('d')
plt.ylabel('$EVI$ [u]')
plt.ylim([0,8])
plt.grid()
plt.show()

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
