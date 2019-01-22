#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:10:23 2019

@author: marionthomas
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
sns.set()

def Potential(deltaT,S):
    return S*(deltaT)
    
calib_data = np.loadtxt('thermocouple_data1.csv', delimiter=',', usecols=(0,1), unpack=True)


def chi_square_red(x, y, yerr, modelfunc, p_opt):
    numerator = (y - modelfunc(x, *p_opt)) ** 2
    denominator = yerr ** 2
    chi = sum(numerator / denominator)
    chired = chi / (len(y) - len(p_opt))
    return chired
def fitting(function,xdata,ydata,  guess,sigma = 0):
    if type(sigma) is int:
        fit_opt, fit_cov = curve_fit(function,xdata,ydata, p0=guess)
    else:
        fit_opt, fit_cov = curve_fit(function,xdata,ydata, p0=guess, sigma = sigma, absolute_sigma = True)
    return function(xdata,*fit_opt), fit_opt, fit_cov

def Temperature(voltage):
    temp1 = 0
    S = fit[1][0]
    temp2 = temp1 - (voltage/S)
    
    return temp2

plt.figure(figsize = (8,8))

T1 = 0 # 0 °C for reference bath
deltaT = T1 - calib_data[0]

plt.scatter(deltaT,calib_data[1] )
init = (0.007)
errors = np.repeat(0.1, len(calib_data[1]))
fit = fitting(Potential,deltaT,calib_data[1] ,guess =init, sigma = errors)
plt.plot(deltaT, fit[0], c = 'r')
plt.errorbar(deltaT,calib_data[1], yerr = errors, ecolor='g', fmt='o', capthick=2)
plt.title('EMF(V) vs. T', fontsize = 16)
plt.ylabel('Voltage (mV)', fontsize = 14)
plt.xlabel('T1-T2', fontsize = 14)


print('\u03C7^2 = ', chi_square_red(deltaT,calib_data[1], errors,Potential, fit[1]))
perr = np.sqrt(np.diag(fit[2]))
print('S = ', fit[1][0] )
print('error on S = ', perr[0] )

voltage = float(input("What Did the Voltmeter Say (in mV) ? "))
temp_val = Temperature(voltage)
print('Your Temperature ' + np.str(np.round(temp_val,3)) + ' C!')



