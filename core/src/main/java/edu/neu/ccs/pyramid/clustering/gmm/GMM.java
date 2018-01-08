package edu.neu.ccs.pyramid.clustering.gmm;

import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.linear.RealVector;

public class GMM {
    private int numComponents;
    private GaussianDistribution[] gaussianDistributions;
    private double[] mixtureCoefficients;

    public int getNumComponents() {
        return numComponents;
    }

    public GaussianDistribution[] getGaussianDistributions() {
        return gaussianDistributions;
    }

    public double[] getMixtureCoefficients() {
        return mixtureCoefficients;
    }

    private double logDensity(RealVector instance){
        double[] arr = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            arr[k]=Math.log(mixtureCoefficients[k])+gaussianDistributions[k].logDensity(instance);
        }
        return MathUtil.logSumExp(arr);
    }

     double[] posteriors(RealVector instance){
        double[] arr = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            arr[k]=Math.log(mixtureCoefficients[k])+gaussianDistributions[k].logDensity(instance);
        }
        double[] posteriors = new double[numComponents];
        double logDenominator = MathUtil.logSumExp(arr);
        for (int k=0;k<numComponents;k++){
            posteriors[k] = Math.exp(arr[k]-logDenominator);
        }
        return posteriors;
    }


}
