package edu.neu.ccs.pyramid.clustering.bm;

import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomGeneratorFactory;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
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
public class BM implements Serializable{
    private static final long serialVersionUID = 2L;
    private int numClusters;
    private int dimension;
    /**
     * format:[cluster][dimension]
     */
    BernoulliDistribution[][] distributions;
    double[] mixtureCoefficients;
    double[] logMixtureCoefficients;
    // the log probability of the empty vector in each cluster
    // size = num clusters
    private double[] logClusterConditioinalForEmpty;
    private List<String> names;

    public BM(int numClusters, int dimension, long randomSeed) {
        this.numClusters = numClusters;
        this.dimension = dimension;
        this.distributions = new BernoulliDistribution[numClusters][dimension];
        this.mixtureCoefficients = new double[numClusters];
        Arrays.fill(mixtureCoefficients,1.0/numClusters);
        this.logMixtureCoefficients = new double[numClusters];
        Arrays.fill(logMixtureCoefficients,Math.log(1.0/numClusters));
        Random random = new Random(randomSeed);
        RandomGenerator randomGenerator = RandomGeneratorFactory.createRandomGenerator(random);
        UniformRealDistribution uniform = new UniformRealDistribution(randomGenerator, 0.25,0.75);
        for (int k=0;k<numClusters;k++){
            for (int d=0;d<dimension;d++){
                double p = uniform.sample();
                distributions[k][d] = new BernoulliDistribution(p);
            }
        }
        this.logClusterConditioinalForEmpty = new double[numClusters];
        updateLogClusterConditioinalForEmpty();

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
        double logProb = logClusterConditioinalForEmpty[clusterIndex];
        for (Vector.Element nonzero: vector.nonZeroes()){
            int l = nonzero.index();
            BernoulliDistribution distribution = distributions[clusterIndex][l];
            logProb -= distribution.logProbability(0);
            logProb += distribution.logProbability(1);
        }
        return logProb;
    }

    private double computeLogClusterConditionalForEmpty(int clusterIndex){
        double logProb = 0.0;
        for (int l=0;l< dimension;l++){
            BernoulliDistribution distribution = distributions[clusterIndex][l];
            logProb += distribution.logProbability(0);
        }
        return logProb;
    }

    void updateLogClusterConditioinalForEmpty(){
        IntStream.range(0, numClusters)
                .forEach(k->logClusterConditioinalForEmpty[k] = computeLogClusterConditionalForEmpty(k));
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
     * log probability based on the mixture
     * @param vector
     * @return
     */
    public double logProbability(Vector vector){
        double[] clusterConditionalLogProbArr = clusterConditionalLogProbArr(vector);

        double[] arr = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            arr[k] = logMixtureCoefficients[k]+clusterConditionalLogProbArr[k];
        }

        return MathUtil.logSumExp(arr);
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

    public BernoulliDistribution[][] getDistributions() {
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
        int[] sortedComponents = ArgSort.argSortDescending(mixtureCoefficients);
        for (int k:sortedComponents){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
            sb.append("probabilities = ").append("[");
            List<Pair<String,Double>> pairs = new ArrayList<>();
            for (int d=0;d<dimension;d++){
                Pair<String,Double> pair = new Pair<>(names.get(d),distributions[k][d].getP());
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
