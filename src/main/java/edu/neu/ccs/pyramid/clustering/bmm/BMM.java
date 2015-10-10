package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Bernoulli mixture model for binary feature valued dataset clustering
 * no label is considered
 * fixed mixture coefficients are used
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
    List<String> names;

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
        this.names = new ArrayList<>(dimension);
        for (int d=0;d<dimension;d++){
            names.add(""+d);
        }
    }

    public void setNames(List<String> names) {
        this.names = names;
    }

    public List<String> getNames() {
        return this.names;
    }


    public double clusterConditionalLogProb(Vector vector, int clusterIndex){
        double prob = 0.0;
        for (int l=0;l< dimension;l++){
            BinomialDistribution distribution = distributions[clusterIndex][l];
            prob += Math.log(distribution.probability((int)vector.get(l)));
        }
        return prob;
    }

    /**
     * return the clusterConditionalLogProb for each cluster.
     * @param vector
     * @return
     */
    public double[] clusterConditionalLogProbArr(Vector vector){
        double[] probArr = new double[numClusters];

        for (int clusterIndex=0; clusterIndex<numClusters; clusterIndex++) {
            probArr[clusterIndex] = clusterConditionalLogProb(vector, clusterIndex);
        }
        return probArr;
    }

    /**
     * sample a vector from the mixture distribution
     * @return
     */
    public Vector sample(){
        Vector vector = new DenseVector(dimension);
        // first sample cluster
        int[] clusters = IntStream.range(0,numClusters).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,mixtureCoefficients);
        int cluster = enumeratedIntegerDistribution.sample();
        // then sample each dimension
        for (int d=0;d<dimension;d++){
            vector.set(d,distributions[cluster][d].sample());
        }
        return vector;
    }

    /**
     * sample a vector by the k-th single distribution.
     * @return Vector
     */
    public Vector sample(int kCluster) {
        if ((kCluster<0) || (kCluster>=numClusters)) {
            throw new RuntimeException("Please given a legal k-th cluster");
        }
        Vector vector = new DenseVector(dimension);
        for (int d=0;d<dimension;d++){
            vector.set(d,distributions[kCluster][d].sample());
        }
        return vector;
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
        sb.append(", dimension=").append(dimension).append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
            sb.append("probabilities = ").append("[");
            List<Pair<String,Double>> pairs = new ArrayList<>();
            for (int d=0;d<dimension;d++){
                Pair<String,Double> pair = new Pair<>(names.get(d),distributions[k][d].getProbabilityOfSuccess());
                pairs.add(pair);
//                sb.append(names.get(d)).append(":").append(distributions[k][d].getProbabilityOfSuccess());
//                if (d!=dimension-1){
//                    sb.append(", ");
//                }
            }
            Comparator<Pair<String,Double>> comparator = Comparator.comparing(Pair::getSecond);
            List<Pair<String,Double>> sorted = pairs.stream().sorted(comparator.reversed())
                    .collect(Collectors.toList());
            for (int d=0;d<dimension;d++){
                Pair<String,Double> pair = sorted.get(d);
                sb.append(pair.getFirst()).append(":").append(pair.getSecond());
                    if (d!=dimension-1){
                    sb.append(", ");
                }
            }
            sb.append("]");
            sb.append("\n");
        }

        sb.append('}');
        return sb.toString();
    }
}
