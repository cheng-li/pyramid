package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by chengli on 11/3/15.
 */
public class GreedyInitializer {
    int numClusters;
    // format [#data][#cluster+1]
    double[][] gammasAllClusters;
    // format [#cluster+1][#data]
    double[][] gammasAllClustersT;
    MultiLabelClfDataSet dataSet;
    // format [#labels][#data][2]
    private double[][][] targetsDistributions;
    private double priorVariance;
    // format [#cluster][#label]
    private LogisticRegression[][] logisticRegressions;
    // format [#data]
    private Vector[] labels;

    public GreedyInitializer(MultiLabelClfDataSet dataSet, int numClusters, double variance) {
        this.dataSet = dataSet;
        this.priorVariance = variance;
        int numClasses = dataSet.getNumClasses();

        this.numClusters = numClusters;
        // initialize distributions
        this.logisticRegressions = new LogisticRegression[numClusters][numClasses];

        this.labels = new DenseVector[dataSet.getNumDataPoints()];
        for (int n=0; n<labels.length; n++) {
            Set<Integer> label = dataSet.getMultiLabels()[n].getMatchedLabels();
            labels[n] = new DenseVector(dataSet.getNumClasses());
            for (int l : label) {
                labels[n].set(l, 1);
            }
        }

        this.targetsDistributions = new double[numClasses][dataSet.getNumDataPoints()][2];
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            Vector label = labels[n];
            for (int l=0; l<label.size(); l++) {
                if (label.get(l) == 0.0) {
                    this.targetsDistributions[l][n][0] = 1;
                } else {
                    this.targetsDistributions[l][n][1] = 1;
                }
            }
        }

        this.gammasAllClusters = new double[dataSet.getNumDataPoints()][numClusters+1];
        this.gammasAllClustersT = new double[numClusters+1][dataSet.getNumDataPoints()];
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            gammasAllClusters[n][0] = 1;
            gammasAllClustersT[0][n] = 1;
        }
    }

    void train(){
        for (int k=0;k<numClusters;k++){
            train(k);
            updateGammas(k);
        }
    }


    void train(int clusterIndex, int classIndex){
        double[] gammas = gammasAllClustersT[clusterIndex];
        LogisticRegression logisticRegression = new LogisticRegression(2,dataSet.getNumFeatures());
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticRegression,
                dataSet, gammas, targetsDistributions[classIndex], priorVariance);
        ridgeLogisticOptimizer.optimize();
        logisticRegressions[clusterIndex][classIndex] = logisticRegression;
    }


    void train(int clusterIndex){
        IntStream.range(0,dataSet.getNumClasses()).parallel()
                .forEach(l -> train(clusterIndex,l));
    }

    private double clusterConditionalLogProb(Vector X, MultiLabel Y, int k) {
        LogisticRegression[] logisticRegressionsK = logisticRegressions[k];

        double logProbResult = 0.0;
        for (int l=0; l<logisticRegressionsK.length; l++) {
            double[] logProbs = logisticRegressionsK[l].predictClassLogProbs(X);
            if (Y.matchClass(l)) {
                logProbResult += logProbs[1];
            } else {
                logProbResult += logProbs[0];
            }
        }
        return logProbResult;
    }

    private void updateGammas(int dataPoint, int currentCluster){
        double currentGamma = gammasAllClustersT[currentCluster][dataPoint];
        if (currentGamma==1){
            double logProb = clusterConditionalLogProb(dataSet.getRow(dataPoint),dataSet.getMultiLabels()[dataPoint],currentCluster);
            if (logProb<Math.log(0.6)){
                gammasAllClusters[dataPoint][currentCluster] = 0;
                gammasAllClustersT[currentCluster][dataPoint] = 0;
                gammasAllClusters[dataPoint][currentCluster+1] = 1;
                gammasAllClustersT[currentCluster+1][dataPoint] = 1;
            }
        }
    }

    private void updateGammas(int currentCluster){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(n -> updateGammas(n,currentCluster));
    }

    private void updateGammasLeft(){

    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("GreedyInitializer{").append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster ").append(k).append(", sum of gammas = ").append(MathUtil.arraySum(gammasAllClustersT[k]))
            .append("\n");
        }
        sb.append("gammas left ").append(MathUtil.arraySum(gammasAllClustersT[numClusters]))
                .append("\n");
        sb.append('}');
        return sb.toString();
    }
}
