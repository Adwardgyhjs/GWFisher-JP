import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

base_path = os.path.dirname(os.path.realpath(__file__))

def getMfXfdata(eventname,paraname):
    data=pd.read_table(base_path+'/rin_{}_IMR_{}.dat'.format(eventname,paraname))
    return data


def InvertCov(MXcov):
    if np.linalg.det(MXcov) != 0:
        InvCov=np.linalg.inv(MXcov)
    return InvCov


def CovAndDet(ParaValue):
    MXcov=np.cov(ParaValue)
    MXcovDet=np.linalg.det(MXcov)
    return MXcov,MXcovDet

def plot_pdf(rv,paraVal):
    x_min=np.min(paraVal[0])
    y_min=np.min(paraVal[1])
    x_max=np.max(paraVal[0])
    y_max=np.max(paraVal[1])

    fig=plt.figure()
    ax=fig.add_axes(Axes3D(fig))
    x=np.arange(50,x_max,0.01)
    y=np.arange(y_min,y_max,0.01)
    #a0 = fig.add_subplot(1, 2, 1, label='a0', projection='3d')
    X, Y = np.meshgrid(x,y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    ax.plot_surface(X,Y,rv.pdf(pos),cmap=plt.cm.cool)
    plt.title('Mf-Xf PDF')
    plt.xlabel('Mf')
    plt.ylabel('Xf')
    plt.savefig(base_path+"/Mf_Xf_pdf_3D.png")
    plt.show()
    #plt.savefig(base_path+"/Mf_Xf_pdf.png")

def M_X_PDF(ParaValue,Para_GR,PE,flag):
    print("paraval",np.shape(ParaValue))
    pMean=np.abs(Para_GR+PE)

    cov,covdet=CovAndDet(ParaValue)
    #invcov=InvertCov(cov)
    print("pmean{} cov{}".format(pMean,cov))
    rv=st.multivariate_normal(pMean,cov)

    x_min=np.min(ParaValue[0])
    y_min=np.min(ParaValue[1])
    x_max=np.max(ParaValue[0])
    y_max=np.max(ParaValue[1])

    if flag=='I' or flag=='MR':
        x, y = np.mgrid[57:65:0.001,0.67:0.72:0.001]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        print("rv.pdf min",np.min(rv.pdf(pos)))
        print("rv.pdf max", np.max(rv.pdf(pos)))
        max_prob = np.max(rv.pdf(pos))
        color=['red','blue','green']
        cnum=np.random.randint(0,3)
        CS=plt.contour(x,y,rv.pdf(pos),[max_prob*0.90,max_prob],colors=color[cnum])
        plt.clabel(CS,inline=1,fontsize=10)
        CS.collections[0].set_label('{}'.format(flag))
        plt.scatter(Para_GR[0],Para_GR[1],marker='+')
        plt.annotate('GR',xy=(Para_GR[0],Para_GR[1]))
        plt.colorbar()
        plt.xlabel('Mf')
        plt.ylabel('Xf')
        plt.savefig(base_path+'/contourf_plot_{}.png'.format(flag))
        #plt.show()
    elif flag=='IMR':
        plt.show()
        x, y = np.mgrid[-0.5:0.8:0.001,-0.6:0.5:0.001]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        print("rv.pdf min",np.min(rv.pdf(pos)))
        print("rv.pdf max", np.max(rv.pdf(pos)))
        max_prob = np.max(rv.pdf(pos))
        plt.contour(x,y,rv.pdf(pos),[max_prob*0.9,max_prob])
        plt.scatter(Para_GR[0],Para_GR[1],marker='+')
        plt.annotate('GR',xy=(Para_GR[0],Para_GR[1]))
        plt.colorbar()
        plt.xlabel('Mf')
        plt.ylabel('Xf')
        plt.savefig(base_path+'/contourf_plot_{}.png'.format(flag))
    #plt.show()
    #plot_pdf(rv,ParaValue)
    return rv
