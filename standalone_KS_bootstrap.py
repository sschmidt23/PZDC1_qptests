#April 8, 2018
#Sam Schmidt

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import skgof
import sys,os
from sklearn.utils import resample

def main():
    """
    script to calculate confidence intervals on KS test from
    bootstraps
    inputs:
      PITvec: np array of Probability Integral Transform values
    output:
      bootstrap confidence interval file for KS, and a plot of the KS bootstrap
      values along with a KDE and a Gaussian fit to them to check that sigma
      is sensible
    """


    basepath = "."
    infile = "TESTPITVALS.out"
    outfile = "STANDALONE_KS_BOOTSTRAP_CONF_INTERVAL.out"
    outfp = open(outfile,"w")

    data = np.loadtxt(infile)
    pits = data[:,1]
    bootksvals = []
    print ("read in PITS")
    nboots = 1000
    bootSampleSize = int(len(pits)*0.5) 
    #set bootstrap sample size as half of the full sample
    for k in range(nboots):
        bootpits = resample(pits,n_samples=bootSampleSize,replace=True)
        print ("Bootstrap #%d Gold sample numbe: %d\n"%(k,len(bootpits)))
        ks_result = skgof.ks_test(bootpits,sps.uniform())
        ksval = ks_result.statistic
        print "KS Val: %.6f"%(ksval)
        bootksvals.append(ksval)

    meanks = np.mean(bootksvals)
    confhigh = np.percentile(bootksvals,84.135)
    conflow = np.percentile(bootksvals,15.865)
    sigma = 0.5*(confhigh-conflow)
    print ("mean ks: %.6f   sigma: %.6f\n"%(meanks,sigma))
    outfp.write("mean KS value: %.6f\nsigma: %.6f\n"%(meanks,sigma))
    outfp.close()

    binedges = np.linspace(meanks-4.*sigma,meanks+4.*sigma,75)
    binwidth = binedges[2]-binedges[1]

    fig = plt.figure(figsize=(10,10))
    plt.hist(bootksvals,bins=binedges,label="histogram")
    xarr = np.arange(meanks-4.*sigma,meanks+4.*sigma,binwidth)
    y = sps.norm(loc=meanks,scale=sigma)
    yarr = float(nboots)*binwidth*y.pdf(xarr)
    tmplabel = "Bootstrap Gaussian\nmean:%.6f sigma=%.6f "%(meanks,sigma)
    plt.plot(xarr,yarr,c='r',lw=3,linestyle='--',label=tmplabel)
    kdex = sps.gaussian_kde(bootksvals)
    ykde = float(nboots)*binwidth*kdex(xarr)
    plt.plot(xarr,ykde,c='g',lw=2,linestyle='-',label="Gaussian KDE fit")
    plt.plot([meanks,meanks],[0,1.3*np.amax(yarr)],lw=4,c='k',label="mean KS")
    plt.xlabel("SkyNet KS Bootstraps",fontsize=18)
    plt.ylabel("Number",fontsize=18)
    plt.legend(loc="upper left")
    plt.savefig("testks.jpg",fmt="jpg")
    plt.show()


    outfp.close()
    print ("finished")
if __name__=="__main__":
    main()
