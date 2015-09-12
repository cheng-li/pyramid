package edu.neu.ccs.pyramid.clustering.bmm;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Bernoulli mixture model
 * dimension has to be big enough in order to identify several clusters
 * when dimension = 1; we can only identify 1 cluster
 * Created by chengli on 9/12/15.
 */
public class BMM {
    private int numClusters;
    private int dimension;
    /**
     * format:[cluster][dimension]
     */
    BinomialDistribution[][] distributions;
    double[] mixtureCoefficients;

    public BMM(int numClusters, int dimension) {
        this.numClusters = numClusters;
        this.dimension = dimension;
        this.distributions = new BinomialDistribution[numClusters][dimension];
        this.mixtureCoefficients = new double[numClusters];
        Arrays.fill(mixtureCoefficients,1.0/numClusters);
        UniformRealDistribution uniform = new UniformRealDistribution(0.25,0.75);
        for (int k=0;k<numClusters;k++){
            for (int d=0;d<dimension;d++){
                double p = uniform.sample();
                distributions[k][d] = new BinomialDistribution(1,p);
            }
        }
    }


    //todo stable?
    double probability(Vector vector, int clusterIndex){
        double prob = 1;
        for (int d=0;d<dimension;d++){
            BinomialDistribution distribution = distributions[clusterIndex][d];
            prob *= distribution.probability((int)vector.get(d));
        }
        return prob;
    }

    public int getNumClusters() {
        return numClusters;
    }

    public int getDimension() {
        return dimension;
    }

    public BinomialDistribution[][] getDistributions() {
        return distributions;
    }

    public double[] getMixtureCoefficients() {
        return mixtureCoefficients;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("BMM{");
        sb.append("numClusters=").append(numClusters);
        sb.append(", dimension=").append(dimension);
        sb.append(", distributions=").append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster "+k).append("\n");
            for (int d=0;d<dimension;d++){
                sb.append(distributions[k][d].getProbabilityOfSuccess()).append(",");
            }
            sb.append("\n");
        }

        sb.append(", mixtureCoefficients=").append(Arrays.toString(mixtureCoefficients));
        sb.append('}');
        return sb.toString();
    }
}
