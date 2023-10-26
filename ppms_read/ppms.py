import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import os

class PPMS(object):
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.type = "RT"

    def extract_RT(self, bridge, T=(2,4), iloc=(0,-1)):
        df=pd.read_csv(self.path, sep=',',header =31)
        if iloc[1]<0:
            df=df.iloc[iloc[0]:]
        else:
            df=df.iloc[iloc[0]:iloc[1]]


        t_str = 'Temperature (K)'
        df = df[df[t_str]>=T[0]]
        df = df[df[t_str]<=T[1]]


        R_ret = df['Bridge {} Resistance (Ohms)'.format(bridge)]
        T_ret = df[t_str]
        self.type = "RT"
        self.data = pd.DataFrame({'R': R_ret, 'T': T_ret})
        return 0

    def extract_Ic(self, bridge, iloc=(0,-1)):
        df=pd.read_csv(self.path, sep=',',header =31)
        if iloc[1]<0:
            df=df.iloc[iloc[0]:]
        else:
            df=df.iloc[iloc[0]:iloc[1]]

        oe = 'Magnetic Field (Oe)'
        uA = 'Bridge {} Excitation (uA)'.format(bridge)
        ohm = 'Bridge {} Resistance (Ohms)'.format(bridge)

        df = df[df[oe].notnull()]
        df[oe] = list(map(int,df[oe]))



        uA_plus = mk_data(df[df[uA]>0], x=uA, cond=oe, basis=ohm, ascending=True)
        uA_minus = mk_data(df[df[uA]<0], x=uA, cond=oe, basis=ohm, ascending=False)
        Oe = list(set(df[oe]))
        Oe.sort()

        self.type = "Ic"
        self.data = pd.DataFrame({'Oe': Oe, 'uA_plus': uA_plus, 'uA_minus': uA_minus})
        return 0

    def extract_Hc(self, bridge, I,  iloc=(0,-1)):
        df=pd.read_csv(self.path, sep=',',header =31)
        if iloc[1]<0:
            df=df.iloc[iloc[0]:]
        else:
            df=df.iloc[iloc[0]:iloc[1]]

        oe = 'Magnetic Field (Oe)'
        uA = 'Bridge {} Excitation (uA)'.format(bridge)
        ohm = 'Bridge {} Resistance (Ohms)'.format(bridge)
        T = 'Temperature (K)'

        df = df[df[oe].notnull()]
        df[oe] = list(map(int,df[oe]))

        df = df[df[uA]>=I[0]]
        df = df[df[uA]<=I[1]]

        df = df[df[T].notnull()]
        df[T] = list(map(lambda l:
                         Decimal(str(l)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                         , df[T]))

        H = mk_data(df, x=oe, cond=T, basis=ohm, ascending=True)
        temperature = list(set(df[T]))
        temperature.sort()

        self.type = "Hc"
        self.data = pd.DataFrame({'T': temperature, 'H': H})

        return 0




    def mk_fig(self):
        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 'xx-large',
                 'axes.labelsize': 'xx-large',
                 'axes.titlesize':'xx-large',
                 'xtick.labelsize':'xx-large',
                 'ytick.labelsize':'xx-large'}
        pylab.rcParams.update(params)
        axisfontsize=18
        df = self.data
        if self.type=="Ic":
            mk_fig_Ic(df)
        elif self.type=="Hc":
            mk_fig_Hc(df)
        elif self.type=="RT":
            mk_fig_RT(df)

def mk_fig_Ic(df):
    axisfontsize=18
    df["uA_delta"] = df["uA_plus"] + df["uA_minus"]
    df["eta"] = 100*df["uA_delta"]/(df["uA_plus"]-df["uA_minus"])
    plt.scatter(df["Oe"], df["uA_delta"])
    plt.xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)

    plt.show()
    plt.clf()
    plt.scatter(df["Oe"], df["eta"])
    plt.xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)
    plt.ylabel(r'$\eta$[%]', fontsize=axisfontsize)
    plt.show()
    plt.clf()
    plt.scatter(df["Oe"], df["uA_plus"], label=r'$I_{\plus}$', color='red')
    plt.scatter(df["Oe"], -df["uA_minus"], label=r'$I_{\minus}$', color='blue')
    plt.xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)
    plt.ylabel(r'$I_c\mathrm{[\mu A]}$', fontsize=axisfontsize)
    plt.legend()
    plt.show()

def mk_fig_Hc(df):
    axisfontsize=18
    plt.scatter(df["T"], df["H"])
    plt.xlabel(r"$Temperature\mathrm{[K]}$", fontsize=axisfontsize)
    plt.ylabel(r"$H_c\mathrm{[Oe]}$", fontsize=axisfontsize)

    plt.show()

def mk_fig_RT(df):
    axisfontsize=18
    plt.scatter(df["T"], df["R"])
    plt.xlabel(r"$Temperature\mathrm{[K]}$", fontsize=axisfontsize)
    plt.ylabel(r'$Resistance\mathrm{[\Omega]}$', fontsize=axisfontsize)

    plt.show()

def mk_data(df, x,cond, basis, ascending):
    T = list((set(df[cond])))
    T.sort()
    ret = [None]*len(T)
    minimum_index = 0


    for i,_T in enumerate(T):
        _df = df[df[cond]==_T].sort_values(by=x, ascending=ascending)

        if not _df.empty:
            data = list(_df[basis])

            index = find_index(data, find_threshold(data))
            if index==len(_df):
                print("Exceed Error {}".format(_T))
                print("{}/{}".format(index, len(_df)))
                print("{}: {}_{}".format(list(_df[basis])[-1], list(_df[x])[0],list(_df[x])[-1]))
            elif index<minimum_index:
                print("Small Number Error {}".format(_T))
                print("{}/{}".format(index, len(_df)))
                print("{}: {}_{}".format(list(_df[basis])[-1], list(_df[x])[0],list(_df[x])[-1]))
            else:
                ret[i] = list(_df[x])[index-1]

    return ret

def find_index(data, threshold):
    max_i = 0
    for i, v in enumerate(data):
        if v<threshold and i>max_i:
            max_i = i
    return max_i+1

def find_threshold(data):
    rate = 0.1
    max_i = 0
    max_v = 0
    l = len(data)
    mu = np.mean(data)
    sigma = np.var(data)
    for i in range(1,l-1):
        mu0 = np.mean(data[:i])
        mu1 = np.mean(data[i:])
        sigma_b = (i*(mu0-mu)**2+(l-i)*(mu1-mu)**2)/l
        temp = sigma_b / (sigma-sigma_b)
        if temp>max_v:
            max_i = i
            max_v = temp
    return(rate*np.mean(data[:max_i])+(1-rate)*np.mean(data[max_i:]))
