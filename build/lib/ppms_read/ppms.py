import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import os

class PPMS(object):
    def __init__(self, path):
        self.df = pd.read_csv(path, sep=',',header =31)
        self.data = {}
        self.type = "RT"
        self.fig = []

    def extract_RT(self, bridge, T=(2,4), iloc=(0,-1)):
        df= self.df
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
        df=self.df
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
        df=self.df
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

    def extract_direction(self, bridge, iloc=(0,-1)):
        df = self.df

        if iloc[1]<0:
            df=df.iloc[iloc[0]:]
        else:
            df=df.iloc[iloc[0]:iloc[1]]

        oe = 'Magnetic Field (Oe)'
        uA = 'Bridge {} Excitation (uA)'.format(bridge)
        ohm = 'Bridge {} Resistance (Ohms)'.format(bridge)
        th = 'Sample Position (deg)'
        df = df[df[th].notnull()]
        #print(np.sort(list(set(df[th]))))
        df[th] = list(map(lambda l:
                     (int(Decimal(str(l)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))-90)%360
                     , df[th]))
        df = df.sort_values(th)
        th_set = {l for l in set(df[th]) if (l+180 in set(df[th])) or (l-180 in set(df[th]))}
        df = df[df[th].isin(th_set)].sort_values(th)
        threshold = 180
        uA_plus = np.array(mk_data(df[df[th]<threshold], x=uA, cond=th, basis=ohm, ascending=True))
        uA_minus = np.array(mk_data(df[df[th]>=threshold], x=uA, cond=th, basis=ohm, ascending=True))
        odd = np.concatenate([(uA_plus-uA_minus)/2, -(uA_minus-uA_plus)/2])
        even = np.concatenate([(uA_plus+uA_minus)/2, (uA_minus+uA_plus)/2])
        theta = np.deg2rad(np.sort(list(set(df[th]))))

        self.type = "direction"
        self.data = pd.DataFrame({'odd': odd, 'even': even,'theta': theta})
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
        self.fig = []
        if self.type=="Ic":
            self.__mk_fig_Ic()
        elif self.type=="Hc":
            self.__mk_fig_Hc()
        elif self.type=="RT":
            self.__mk_fig_RT()
        elif self.type=="direction":
            self.__mk_fig_direction()

    def update_fig(self, i=0):
        fig = self.fig[i]
        managed_fig = plt.figure()
        canvas_manager = managed_fig.canvas.manager
        canvas_manager.canvas.figure = fig
        fig.set_canvas(canvas_manager.canvas)
        plt.savefig("Image/update.png", bbox_inches='tight')
        plt.show()

    def __mk_fig_Ic(self):
        axisfontsize=18
        df = self.data
        fig = plt.figure()
        self.fig.append(fig)
        ax = fig.add_subplot()
        df["uA_delta"] = df["uA_plus"] + df["uA_minus"]
        df["eta"] = 100*df["uA_delta"]/(df["uA_plus"]-df["uA_minus"])
        ax.scatter(df["Oe"], df["uA_delta"])
        ax.set_xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)
        #plt.ylim([0,30])


        plt.show()
        plt.clf()
        fig = plt.figure()
        self.fig.append(fig)
        ax = fig.add_subplot()
        ax.scatter(df["Oe"], df["eta"])
        ax.set_xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)
        ax.set_ylabel(r'$\eta$[%]', fontsize=axisfontsize)
        #plt.ylim([0,3.7])
        plt.savefig("Image/eta.png", bbox_inches='tight')
        plt.show()
        plt.clf()
        fig = plt.figure()
        self.fig.append(fig)
        ax = fig.add_subplot()
        ax.scatter(df["Oe"], df["uA_plus"], label=r'$I_{\plus}$', color='red')
        ax.scatter(df["Oe"], -df["uA_minus"], label=r'$I_{\minus}$', color='blue')
        ax.set_xlabel(r"$H\mathrm{[Oe]}$", fontsize=axisfontsize)
        ax.set_ylabel(r'$I_c\mathrm{[\mu A]}$', fontsize=axisfontsize)
        ax.legend()
        #plt.ylim([0,420])
        plt.savefig("Image/Ic.png", bbox_inches='tight')
        plt.show()

    def __mk_fig_Hc(self):
        axisfontsize=18
        df = self.data
        fig = plt.figure()
        self.fig.append(fig)
        ax = fig.add_subplot()
        ax.scatter(df["T"], df["H"])
        ax.set_xlabel(r"$Temperature\mathrm{[K]}$", fontsize=axisfontsize)
        ax.set_ylabel(r"$H_c\mathrm{[Oe]}$", fontsize=axisfontsize)

        plt.show()

    def __mk_fig_RT(self):
        axisfontsize=18
        df = self.data
        fig = plt.figure()
        self.fig.append(fig)
        ax = fig.add_subplot()
        ax.scatter(df["T"], df["R"])
        ax.plot(df["T"], df["R"])
        ax.set_xlabel(r"$Temperature\mathrm{[K]}$", fontsize=axisfontsize)
        ax.set_ylabel(r'$Resistance\mathrm{[\Omega]}$', fontsize=axisfontsize)
        plt.savefig("Image/RT.png", bbox_inches='tight')

        plt.show()

    def __mk_fig_direction(self):
        axisfontsize=18
        df = self.data

        fig = plt.figure()
        self.fig.append(fig)
        theta = df['theta']
        even = df['even']
        ax = fig.add_subplot(projection='polar')
        ax.scatter(theta, even)
        ax.plot(theta, even)
        plt.savefig("Image/direction_even.png", bbox_inches='tight')
        plt.show()
        plt.clf()

        fig = plt.figure()
        self.fig.append(fig)
        odd = df['odd']
        ax = fig.add_subplot(projection='polar')
        ax.scatter(theta, odd)
        ax.plot(theta, odd)
        plt.savefig("Image/direction_odd.png", bbox_inches='tight')
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
