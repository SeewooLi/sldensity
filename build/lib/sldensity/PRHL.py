import scipy.special as gms
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

############################################################
# Basic Functions (PDF, CDF, inverse-CDF, random sampling)
############################################################

def dPRHL(x, a, m, l):
    pdf= a*l*np.exp(-l*(x-m))/(1+np.exp(-l*(x-m)))**(a+1)
    return(pdf)

def cPRHL(x, a, m, l):
    cdf= (1+np.exp(-l*(x-m)))**(-a)
    return(cdf)

def qPRHL(q, a, m, l):
    quantile= m-np.log(q**(-1/a)-1)/l
    return(quantile)

def rPRHL(n, a, m, l):
    samples = m-1/l*np.log(np.random.uniform(0, 1, size=n)**(-1/a)-1)
    return(samples)

############################################################
# Transformation of PRHL alpha to Gaussian sigma_0 & sigma_1
############################################################

def SS_a_to_sigma(s,a):
    xx = np.arange(0.01,.5,0.01)
    l = np.sqrt((gms.polygamma(1,a)+gms.polygamma(1,1))/1)
    m = np.log(2**(1/a)-1)/l
    r1 = sum((qPRHL(xx,a,m,l)-norm.ppf(xx,0,s[0]))**2)+sum((qPRHL(1-xx,a,m,l)-norm.ppf(1-xx,0,s[1]))**2)
    return(r1)

def a_to_sigma(a):
    res = minimize(SS_a_to_sigma, x0=[1,1], method = 'Nelder-Mead',args=(a,))
    return(res)


############################################################
# PRHL Parameter Estimation (PRHL and mixture-PRHL)
############################################################

def func(x, sk):
    return((sk*(gms.polygamma(1,x)+gms.polygamma(1,1))**(3/2)-
            (gms.polygamma(2,x)-gms.polygamma(2,1))))

def m_grad(x, sk):
    return((1.5*sk*gms.polygamma(2,x)*(gms.polygamma(1,x)+gms.polygamma(1,1))**(.5)-gms.polygamma(3,x)))

def alpha(sk):
    a = 1
    thres = 1
    while thres >= 0.00001:
        thres = -func(x=a, sk=sk)/m_grad(x=a, sk=sk)
        a += thres  
        thres = abs(thres)      
    return(a)


def alpha_(sk, dim, a=None):
    if np.array(a == None).any():
        a = np.repeat(1, dim)
    thres = 1
    div = 0.5
    while thres >= 0.00001:
        thres = -func(x=a, sk=sk)/m_grad(x=a, sk=sk)
        if abs(thres).max()<div:
            a = a+thres
            div = abs(thres).max()
        else:
            a = a+thres*div/((abs(thres)).max())
        thres = (abs(thres)).max()        
    return(a)

def estPRHL(df):
    # if df.x_axis.unique()==0:
    #     # Parameter estimation for PV0
    #     res = trc_logis(df)
    #     a=1
    #     skew = 0
    #     mean = res.x[0]
    #     var = (res.x[1])**2
    #     m = mean
    #     l = 1.702/res.x[1]
    #     median = mean
    # else:
        # Parameter estimation for PV1-7
    tt = np.sum(df.freq)
    mean = df.x_axis.dot(df.freq)/tt
    df = df.loc[(df['x_axis']>=mean-15)&(df['x_axis']<=mean+15),:]
    tt = np.sum(df.freq)
    mean = df.x_axis.dot(df.freq)/tt
    var = ((df.x_axis-mean)**2).dot(df.freq)/(tt-1)
    skew = (((df.x_axis-mean)/np.sqrt(var))**3).dot(df.freq)/tt
    a = alpha(sk = skew)
    l = np.sqrt((gms.polygamma(1,a)+gms.polygamma(1,1))/var)
    m = mean-((gms.digamma(a)-gms.digamma(1))/l)
    median = m-np.log(2**(1/a)-1)/l
    return(a, m, l, mean, var, skew, median)

def moments_df(df):
    tt = np.sum(df.freq)
    mean = df.x_axis.dot(df.freq)/tt
    df = df.loc[(df['x_axis']>=mean-15)&(df['x_axis']<=mean+15),:]
    mean = df.x_axis.dot(df.freq)/tt
    var = 1.2*((df.x_axis-mean)**2).dot(df.freq)/(tt-1)
    skew = (((df.x_axis-mean)/np.sqrt(var))**3).dot(df.freq)/tt
    return(mean, var, skew)

def moments_pdf(a, m, l):
    mean = m+(gms.digamma(a)-gms.digamma(1))/l
    var = (gms.polygamma(1,a)+gms.polygamma(1,1))/(l**2)
    skew = (gms.polygamma(2,a)-gms.polygamma(2,1))/((gms.polygamma(1,a)+gms.polygamma(1,1))**1.5)
    return(mean, var, skew)

def dMPRHL(x, a, m, l):
    pdf = np.zeros(len(x))
    for i in range(7):
        pdf += dPRHL(x, a[i], m[i], l[i])/7
    return(pdf)

def estMPRHL(df, a, m, l):
    idx = np.array(df.x_axis)
    cnt = np.array(df.freq)
    
    n_iters = 0
    thres = 10
    thres2 = 1
    a_n = None

    while (thres >= 0.0001):
        n_iters += 1
        resp = np.empty((0,len(idx)))
        for i in range(0,7,1):
            resp = np.vstack((resp,dPRHL(idx, a=a[i], m=m[i], l=l[i])))   
        resp = resp/(resp.sum(axis=0))*np.tile(cnt,reps=(7,1))
        resp[np.isnan(resp)] = 0
        resp_sum = resp.sum(axis=1)
        
        # mean, variance, skewness
        ms = resp.dot(idx)/resp_sum
        vs = resp.dot((np.tile(idx, reps=(7,1)).transpose()-ms)**2).diagonal()/resp_sum
        if thres2 <= 0.0001:
            sk = resp.dot(((np.tile(idx, reps=(7,1)).transpose()-ms)/np.sqrt(vs))**3).diagonal()/resp_sum
            a_n = alpha_(sk, dim=7, a = a_n)
            l_n = np.sqrt((gms.polygamma(1,a_n)+gms.polygamma(1,1))/vs)
            m_n = ms-((gms.digamma(a_n)-gms.digamma(1))/l_n)
            thres = np.append(abs(a_n-a),np.append(abs(m_n-m),abs(l_n-l))).max()
            a = a_n
            l = l_n
            m = m_n
        else:
            sk=0
            a_n = alpha_(sk, dim=7, a = a_n)
            l_n = np.sqrt((gms.polygamma(1,a)+gms.polygamma(1,1))/vs)
            m_n = ms-((gms.digamma(a)-gms.digamma(1))/l)
            thres2 = np.append(abs(m_n-m),abs(l_n-l)).max()
            a = a_n
            m = m_n
            l = l_n
    
    return(a, m, l, sk)


############################################################
# PV0 Estimation (truncated logistic distribution)
#############################################################

def qlog(q,m,s):
    x = m+s/1.702*np.log(q/(1-q))
    return(x)

def df0(df):
    df_ = np.vstack((df.x_axis.iloc[5:75],(df.freq.cumsum()/df.freq.sum()).iloc[5:75]))
    return(df_)
                    
def SS_trc_logis(par, df):
    df = df[:,(df[1]<=.95)&(df[1]>=.05)]
    x_ = qlog(df[1],par[0],par[1])
    err = np.nansum(abs(df[0]-x_))
    return(err)

def trc_logis(df):
    df = df0(df)
    res = minimize(SS_trc_logis, x0=[5,4], method = 'Powell',args=(df,), bounds=([-15,50],[1,80]))
    return(res)

############################################################
# Read Bias
#############################################################

def PRHL_read_bias(df):
    df = df.reset_index(drop=True)
    x = np.arange(25,170,.1)
    rb = np.append(np.nan, np.zeros(7))
    for i in range(7):
        diff = np.log(dPRHL(x, df.loc[i,'alpha'], df.loc[i,'mu'], df.loc[i,'lambda'])/dPRHL(x, df.loc[i+1,'alpha'], df.loc[i+1,'mu'], df.loc[i+1,'lambda']))
        pos_indices = np.where(diff > 0)
        neg_indices = np.where(diff < 0)
        rb[i+1] = np.intersect1d((neg_indices[0]-1), pos_indices)/10+25
    return(rb)