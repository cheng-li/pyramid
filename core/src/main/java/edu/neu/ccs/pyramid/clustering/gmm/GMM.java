package edu.neu.ccs.pyramid.clustering.gmm;

import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;

public class GMM {
    private int numComponents;
    private GaussianDistribution[] gaussianDistributions;
    private double[] mixtureCoefficients;

    public GMM(int dimension, int numComponents) {
        this.numComponents = numComponents;
        this.mixtureCoefficients = new double[numComponents];
        Arrays.fill(this.mixtureCoefficients,1.0/numComponents);
        this.gaussianDistributions = new GaussianDistribution[numComponents];
        for (int k=0;k<numComponents;k++){
            RealVector mean = new ArrayRealVector(dimension);
            for (int d=0;d<dimension;d++){
                mean.setEntry(d,Math.random());
            }
            RealMatrix cov = new Array2DRowRealMatrix(dimension,dimension);
            for (int d=0;d<dimension;d++){
                cov.setEntry(d,d,1);
            }
            gaussianDistributions[k] = new GaussianDistribution(mean, cov);
        }
    }

    public void setMixtureCoefficient(int k, double value){
        mixtureCoefficients[k] = value;
    }

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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("GMM{");
        sb.append("numComponents=").append(numComponents);
        sb.append(", gaussianDistributions=").append(Arrays.toString(gaussianDistributions));
        sb.append(", mixtureCoefficients=").append(Arrays.toString(mixtureCoefficients));
        sb.append('}');
        return sb.toString();
    }
}
