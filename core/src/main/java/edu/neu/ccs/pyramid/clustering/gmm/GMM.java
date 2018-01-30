package edu.neu.ccs.pyramid.clustering.gmm;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.io.Serializable;
import java.util.Arrays;

public class GMM implements Serializable{
    private static final long serialVersionUID = 1L;
    private int numComponents;
    private GaussianDistribution[] gaussianDistributions;
    private double[] mixtureCoefficients;

    public GMM(int dimension, int numComponents, RealMatrix data) {
        this.numComponents = numComponents;
        this.mixtureCoefficients = new double[numComponents];
        Arrays.fill(this.mixtureCoefficients,1.0/numComponents);
        this.gaussianDistributions = new GaussianDistribution[numComponents];
        double[] mins = new double[data.getColumnDimension()];
        double[] maxs = new double[data.getColumnDimension()];
        double[] vars = new double[data.getColumnDimension()];
        for (int j=0;j<data.getColumnDimension();j++){
            RealVector column = data.getColumnVector(j);
            mins[j] = column.getMinValue();
            maxs[j] = column.getMaxValue();
            DescriptiveStatistics stats = new DescriptiveStatistics(column.toArray());
            vars[j] = stats.getVariance();
        }
        System.out.println("mins = "+Arrays.toString(mins));
        System.out.println("maxs = "+Arrays.toString(maxs));
        for (int k=0;k<numComponents;k++){
            int randomMeanInstance = Sampling.intUniform(0,data.getRowDimension());

            RealVector mean = data.getRowVector(randomMeanInstance).copy();
//            RealVector mean = new ArrayRealVector(dimension);
//            for (int d=0;d<dimension;d++){
//                mean.setEntry(d, Sampling.doubleUniform(mins[d],maxs[d]));
//            }

            RealMatrix cov = new Array2DRowRealMatrix(dimension,dimension);
            for (int d=0;d<dimension;d++){
                cov.setEntry(d,d,vars[d]+0.00001);
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

    public double logDensity(RealVector instance){
        double[] arr = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            arr[k]=Math.log(mixtureCoefficients[k])+gaussianDistributions[k].logDensity(instance);
        }
        return MathUtil.logSumExp(arr);
    }

    public double[] posteriors(RealVector instance){
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
        sb.append("numComponents=").append(numComponents).append("\n");
        sb.append("mixture coefficients = "+Arrays.toString(mixtureCoefficients)).append("\n");
        for (int k=0;k<numComponents;k++){
            sb.append("component "+k).append("\n");
            sb.append("mixture coefficient = "+mixtureCoefficients[k]).append("\n");
            sb.append("mean = "+gaussianDistributions[k].getMean()).append("\n");
            sb.append("covariance = "+gaussianDistributions[k].getCovariance()).append("\n");
        }
        sb.append('}');
        return sb.toString();
    }
}
