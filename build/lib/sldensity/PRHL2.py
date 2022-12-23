import scipy.special as gms
import pandas as pd
import numpy as np
from scipy.stats import norm
def cPRHL(x, a, m, l):
    s= (1+np.exp(-l*(x-m)))**(-a)
    return(s)

def rPRHL(n, a, m, l):
    sss = np.random.uniform(0, 1, size=n)
    return(m-1/l*np.log(sss**(-1/a)-1))

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

def PRHL_param(df):
    tt = np.sum(df.cell_cnt)
    mean = df.bias_idx_num.dot(df.cell_cnt)/tt
    df = df.loc[(df['bias_idx_num']>=mean-15)&(df['bias_idx_num']<=mean+15),:]
    mean = df.bias_idx_num.dot(df.cell_cnt)/tt
    var = 1.2*((df.bias_idx_num-mean)**2).dot(df.cell_cnt)/(tt-1)
    skew = (((df.bias_idx_num-mean)/np.sqrt(var))**3).dot(df.cell_cnt)/tt
    a = alpha(sk = skew)
    l = np.sqrt((gms.polygamma(1,a)+gms.polygamma(1,1))/var)
    m = mean-((gms.digamma(a)-gms.digamma(1))/l)
    return(a,m,l, skew)

def Vt_skew(df):
    mean = df.bias_idx_num.dot(df.cell_cnt)/np.sum(df.cell_cnt)
    df = df.loc[(df['bias_idx_num']>=mean-15)&(df['bias_idx_num']<=mean+15),:]
    mean = df.bias_idx_num.dot(df.cell_cnt)/np.sum(df.cell_cnt)
    var = 1.2*((df.bias_idx_num-mean)**2).dot(df.cell_cnt)/(np.sum(df.cell_cnt)-1)
    skew = (((df.bias_idx_num-mean)/np.sqrt(var))**3).dot(df.cell_cnt)/np.sum(df.cell_cnt)
    return(skew)

def PRHLM(df, a, m, l):
    idx = np.array(df.iloc[:,0])
    cnt = np.array(df.iloc[:,1])
    #ms = len(idx)/8*np.array(range(1,8,1))
    #sds = np.array(5).repeat(7)
    
    #sk = np.zeros(7)
    
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
    
    return(a,m,l, sk)

def dPRHLM(x, a, m, l):
    density = np.zeros(len(x))
    for i in range(7):
        density += dPRHL(x, a[i], m[i], l[i])/7
    return(density)