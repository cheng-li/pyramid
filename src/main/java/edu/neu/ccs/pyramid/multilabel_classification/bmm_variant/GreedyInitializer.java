package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.TrainConfig;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBFactory;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainConfig;
import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
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

    // format [#cluster][#label]
    private Classifier.ProbabilityEstimator[][] probabilityEstimators;


    public GreedyInitializer(MultiLabelClfDataSet dataSet, int numClusters) {
        this.dataSet = dataSet;

        int numClasses = dataSet.getNumClasses();

        this.numClusters = numClusters;
        // initialize distributions
        this.probabilityEstimators = new LKTreeBoost[numClusters][numClasses];


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
            System.out.println(this);
        }
    }


    void train(int clusterIndex, int classIndex){
        System.out.println("training cluster "+clusterIndex+", class "+classIndex+"\n");
        ClfDataSet transformed = prepareData(clusterIndex,classIndex);
        TrainConfig trainConfig = new LKTBTrainConfig()
                .setLearningRate(0.1)
                .setNumLeaves(20)
                .setNumIterations(35);
        LKTreeBoost classifier = (LKTreeBoost)new LKTBFactory().train(transformed,trainConfig);

        probabilityEstimators[clusterIndex][classIndex] = classifier;
    }

    ClfDataSet prepareData(int clusterIndex, int classIndex){
        ClfDataSet full = DataSetUtil.toBinary(this.dataSet,classIndex);
        double[] gammas = gammasAllClustersT[clusterIndex];
        List<Integer> list = new ArrayList<>();
        for (int i=0;i<gammas.length;i++){
            if (gammas[i]==1){
                list.add(i);
            }
        }

        ClfDataSet selected = DataSetUtil.sampleData(full,list);
        return selected;
    }


    void train(int clusterIndex){
        IntStream.range(0,dataSet.getNumClasses())
                .forEach(l -> train(clusterIndex,l));
    }

    private double clusterConditionalLogProb(Vector X, MultiLabel Y, int k) {
        Classifier.ProbabilityEstimator[] estimators = probabilityEstimators[k];

        double logProbResult = 0.0;
        for (int l=0; l<estimators.length; l++) {
            double[] logProbs = estimators[l].predictLogClassProbs(X);
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
            if (logProb<Math.log(0.5)){
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
