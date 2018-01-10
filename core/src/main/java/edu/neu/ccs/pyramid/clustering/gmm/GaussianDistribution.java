package edu.neu.ccs.pyramid.clustering.gmm;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class GaussianDistribution {
    private RealVector mean;
    private RealMatrix covariance;
    private RealMatrix inverseCovariance;
    private double detCovariance;

    public GaussianDistribution(RealVector mean, RealMatrix covariance) {
        this.mean = mean;
        this.covariance = covariance;
        LUDecomposition decomposition = new LUDecomposition(covariance);
        this.inverseCovariance = decomposition.getSolver().getInverse();
        this.detCovariance = decomposition.getDeterminant();
    }


    public RealVector getMean() {
        return mean;
    }

    public void setMean(RealVector mean) {
        this.mean = mean;
    }

    public RealMatrix getCovariance() {
        return covariance;
    }

    public void setCovariance(RealMatrix covariance) {
        this.covariance = covariance;
        LUDecomposition decomposition = new LUDecomposition(covariance);
        this.inverseCovariance = decomposition.getSolver().getInverse();
        this.detCovariance = decomposition.getDeterminant();
    }

    public double logDensity(RealVector x){
        RealVector diff = x.subtract(mean);
        int dim = mean.getDimension();
        return -0.5*dim*Math.log(2*Math.PI)-0.5*Math.log(detCovariance)
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
