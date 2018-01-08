package edu.neu.ccs.pyramid.clustering.gmm;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.stream.IntStream;

public class GMMTrainer {
    private RealMatrix data;
    private double[][] gammas;
    private GMM gmm;

    private void eStep(){
        IntStream.range(0,data.getRowDimension()).parallel()
                .forEach(i->gammas[i]=gmm.posteriors(data.getRowVector(i)));
    }

    private void mStep(){
        IntStream.range(0,gmm.getNumComponents()).parallel()
                .forEach(k->{
                    gmm.getGaussianDistributions()[k].setMean(computeMean(k));
                    gmm.getGaussianDistributions()[k].setCovariance(computeCov(k));
                });
    }

    private RealVector computeMean(int k){
        RealVector res = new ArrayRealVector(data.getColumnDimension());
        double totalGamma = 0;
        for (int i=0;i<data.getRowDimension();i++){
            res = res.add(data.getRowVector(i).mapMultiply(gammas[i][k]));
            totalGamma += gammas[i][k];
        }
        return res.mapDivide(totalGamma);
    }

    private RealMatrix computeCov(int k){
        RealMatrix res = new Array2DRowRealMatrix(data.getColumnDimension(),data.getColumnDimension());
        double totalGamma = 0;
        for (int i=0;i<data.getRowDimension();i++){
            res = res.add(data.getRowVector(i).outerProduct(data.getRowVector(i)).scalarMultiply(gammas[i][k]));
            totalGamma += gammas[i][k];
        }
        RealVector mean = gmm.getGaussianDistributions()[k].getMean();
        return res.scalarMultiply(1/totalGamma).subtract(mean.outerProduct(mean));
    }
}
