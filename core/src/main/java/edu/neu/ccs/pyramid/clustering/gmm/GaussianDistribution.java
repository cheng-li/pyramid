package edu.neu.ccs.pyramid.clustering.gmm;

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.Serializable;

public class GaussianDistribution implements Serializable{
    private static final long serialVersionUID = 1L;
    private RealVector mean;
    private RealMatrix covariance;
    private RealMatrix inverseCovariance;
    // calculating determinant directly results in overflow or underflow
    // we should compute log determinant instead
    // use the method described in http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
    private double logDeterminant;

    public GaussianDistribution(RealVector mean, RealMatrix covariance) {
        this.mean = mean;
        this.setCovariance(covariance);
    }


    public double getLogDeterminant() {
        return logDeterminant;
    }

    public RealVector getMean() {
        return mean;
    }

    void setMean(RealVector mean) {
        this.mean = mean;
    }

    public RealMatrix getCovariance() {
        return covariance;
    }

    public RealMatrix getInverseCovariance() {
        return inverseCovariance;
    }

    void setCovariance(RealMatrix covariance) {
        this.covariance = covariance;
        CholeskyDecomposition decomposition = new CholeskyDecomposition(covariance);
        this.inverseCovariance = decomposition.getSolver().getInverse();
        RealMatrix lMatrix = decomposition.getL();
        double sum = 0;
        for (int i=0;i<lMatrix.getRowDimension();i++){
            sum += Math.log(lMatrix.getEntry(i,i));
        }
        this.logDeterminant = 2*sum;
    }

    public double logDensity(RealVector x){
        RealVector diff = x.subtract(mean);
        int dim = mean.getDimension();
        return -0.5*dim*Math.log(2*Math.PI)-0.5*logDeterminant
                -0.5*(inverseCovariance.preMultiply(diff).dotProduct(diff));

    }

   private double density(RealVector x){
        return Math.exp(logDensity(x));
   }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("GaussianDistribution{");
        sb.append("mean=").append(mean);
        sb.append(", covariance=").append(covariance);
        sb.append('}');
        return sb.toString();
    }
}
