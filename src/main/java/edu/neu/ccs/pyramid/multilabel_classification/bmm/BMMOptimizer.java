package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.*;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMOptimizer {
    private BMMClassifier bmmClassifier;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;
    double[][] gammas;
    // big variance means small regularization
    private double gaussianPriorVariance;

    private Vector[] labels;

    public BMMOptimizer(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet,
                        double gaussianPriorVariance) {
        this.bmmClassifier = bmmClassifier;
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.terminator = new Terminator();
        this.gammas = new double[dataSet.getNumDataPoints()][bmmClassifier.numClusters];

        this.labels = new DenseVector[dataSet.getNumDataPoints()];
        for (int n=0; n<labels.length; n++) {
            Set<Integer> label = dataSet.getMultiLabels()[n].getMatchedLabels();
            labels[n] = new DenseVector(dataSet.getNumClasses());
            for (int l : label) {
                labels[n].set(l, 1);
            }
        }
    }

    public void optimize(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }


    public void iterate(){
        eStep();
        mStep();
        this.terminator.add(getObjective());
    }

    private void eStep(){
        int K = bmmClassifier.numClusters;

        for (int n=0; n<gammas.length; n++) {
            Vector Xn = dataSet.getRow(n);
            Vector Yn = this.labels[n];

            double[] pZnKArr = bmmClassifier.logisticRegression.predictClassProbs(Xn);
            double[] pYnKArr = bmmClassifier.clusterConditionalProbArr(Yn);
            double[] pZnkYnkArr = new double[K];
            double denominator = 0.0;
            for (int k=0; k<K; k++) {
                pZnkYnkArr[k] = pZnKArr[k] * pYnKArr[k];
                denominator += pZnkYnkArr[k];
            }

            for (int k=0; k<K; k++) {
                gammas[n][k] = pZnkYnkArr[k] / denominator;
            }
        }
    }


    private void mStep(){
        updateBernoullis();
        updateLogisticRegression();
    }


    private double getObjective(){
        return 0;
    }

    private void updateBernoullis(){
        IntStream.range(0, bmmClassifier.numClusters).parallel().forEach(this::updateBernoulli);
    }

    private void updateBernoulli(int k) {
        double nk = 0;
        for (int n=0; n<dataSet.getNumDataPoints(); n++){
            nk += gammas[n][k];
        }
        Vector average = new DenseVector(bmmClassifier.getNumClasses());
        for (int n=0; n<dataSet.getNumDataPoints(); n++){
            average = average.plus(labels[n].times(gammas[n][k]));
        }
        average = average.divide(nk);
        for (int l=0; l<bmmClassifier.numLabels; l++){
            bmmClassifier.distributions[k][l] = new BinomialDistribution(1,average.get(l));
        }

    }

    private void updateLogisticRegression(){
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(bmmClassifier.logisticRegression, dataSet, gammas, gaussianPriorVariance);
        ridgeLogisticOptimizer.optimize();
    }
}
