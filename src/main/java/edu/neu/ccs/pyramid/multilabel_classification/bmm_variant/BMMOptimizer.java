package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.logistic_regression.KLLogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.WeightedLogisticLoss;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 10/23/15.
 */
public class BMMOptimizer implements Serializable {
    private static final Logger logger = LogManager.getLogger();
    private BMMClassifier bmmClassifier;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;

    // format [#data][#cluster]
    double[][] gammas;
    // format [#cluster][#data]
    double[][] gammasT;

    // regularization for softMaxRegression
    private double gaussianPriorforSoftMax;
    // regularization for binary logisticRegression
    private double gaussianPriorforLogit;

    // format [#data]
    private Vector[] labels;

    // format [#labels][#data][2]
    private double[][][] targetsDistributions;

    public BMMOptimizer(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet,
                        double gaussianPriorforSoftMax, double gaussianPriorforLogit) {
        this.bmmClassifier = bmmClassifier;
        this.dataSet = dataSet;
        this.gaussianPriorforSoftMax = gaussianPriorforSoftMax;
        this.gaussianPriorforLogit = gaussianPriorforLogit;

        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);

        this.gammas = new double[dataSet.getNumDataPoints()][bmmClassifier.numClusters];

        this.gammasT = new double[bmmClassifier.numClusters][dataSet.getNumDataPoints()];
        for (int n=0;n<dataSet.getNumDataPoints();n++){
            for (int k=0;k<bmmClassifier.numClusters;k++){
                gammas[n][k] = 1.0/bmmClassifier.numClusters;
                gammasT[k][n] = 1.0/bmmClassifier.numClusters;
            }
        }
        this.labels = new DenseVector[dataSet.getNumDataPoints()];
        for (int n=0; n<labels.length; n++) {
            Set<Integer> label = dataSet.getMultiLabels()[n].getMatchedLabels();
            labels[n] = new DenseVector(dataSet.getNumClasses());
            for (int l : label) {
                labels[n].set(l, 1);
            }
        }

        // 2 means binary classification problem.
        this.targetsDistributions = new double[bmmClassifier.getNumClasses()][dataSet.getNumDataPoints()][2];
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
    }

    public void optimize() {
        while (true) {
            iterate();
            if (terminator.shouldTerminate()) {
                break;
            }
        }
    }

    public void iterate() {
        eStep();
        mStep();
        this.terminator.add(getObjective());
    }

    private void eStep(){
        if (logger.isDebugEnabled()){
            logger.debug("start E step");
        }
        updateGamma();
        if (logger.isDebugEnabled()){
            logger.debug("finish E step");
            logger.debug("objective = "+getObjective());
        }
    }

    private void updateGamma() {
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateGamma);
    }

    private void updateGamma(int n) {
        Vector X = dataSet.getRow(n);
        Vector Y = this.labels[n];
        int K = bmmClassifier.numClusters;
        // log[p(z_n=k | x_n)] array
        double[] logLogisticProbs = bmmClassifier.softMaxRegression.predictClassLogProbs(X);
        // log[p(y_n | z_n=k, x_n)] for all k from 1 to K;
        double[] logClusterConditionalProbs = bmmClassifier.clusterConditionalLogProbArr(X, Y);
        double[] logNumerators = new double[logLogisticProbs.length];
        for (int k=0; k<K; k++) {
            logNumerators[k] = logLogisticProbs[k] + logClusterConditionalProbs[k];
        }
        double logDenominator = MathUtil.logSumExp(logNumerators);
        for (int k=0; k<K; k++) {
            double value = Math.exp(logNumerators[k] - logDenominator);
            gammas[n][k] = value;
            gammasT[k][n] = value;
        }
    }

    void mStep() {
        if (logger.isDebugEnabled()){
            logger.debug("start M step");
        }
        updateBinaryLogisticRegressions();
        updateSoftMaxRegression();
        if (logger.isDebugEnabled()){
            logger.debug("finish M step");
            logger.debug("objective = "+getObjective());
        }
    }

    private void updateBinaryLogisticRegressions() {
        //TODO no parallel
        IntStream.range(0, bmmClassifier.numClusters).forEach(this::updateBinaryLogisticRegression);
    }

    private void updateBinaryLogisticRegression(int k) {
        LogisticRegression[] logisticRegressions = bmmClassifier.binaryLogitRegressions[k];

        for (int l=0; l<bmmClassifier.getNumClasses(); l++) {
            RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticRegressions[l],
                    dataSet, gammasT[k], targetsDistributions[l], gaussianPriorforLogit);
            ridgeLogisticOptimizer.optimize();
        }
    }

    private void updateSoftMaxRegression() {
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(bmmClassifier.softMaxRegression,
                dataSet, gammas, gaussianPriorforSoftMax);
        ridgeLogisticOptimizer.optimize();
    }

    public static Object[] getColumn(Object[][] array, int index){
        Object[] column = new Object[array[0].length]; // Here I assume a rectangular 2D array!
        for(int i=0; i<column.length; i++){
            column[i] = array[i][index];
        }
        return column;
    }


    public double getObjective() {
        KLLogisticLoss logisticLoss =  new KLLogisticLoss(bmmClassifier.softMaxRegression,
                dataSet, gammas, gaussianPriorforSoftMax);
        // Q function for \Thata + gamma.entropy and Q function for Weights
        return logisticLoss.getValue() + binaryLogitsObj();
    }

//    private double getMStepObjective() {
//        KLLogisticLoss logisticLoss =  new KLLogisticLoss(bmmClassifier.softMaxRegression,
//                dataSet, gammas, gaussianPriorforSoftMax);
//        // Q function for \Thata + gamma.entropy and Q function for Weights
//        return logisticLoss.getValue() + binaryLogitsObj();
//    }
//
//    private double getEntropy() {
//
//        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
//                .mapToDouble(this::getEntropy).sum();
//    }

//    private double getEntropy(int i) {
//        return Entropy.entropy(gammas[i]);
//    }

    private double binaryLogitsObj() {
        double res = IntStream.range(0,bmmClassifier.numClusters)
                .mapToDouble(this::binaryLogitsObj).sum();
        if (logger.isDebugEnabled()){
            logger.debug("binary logistic objectives = "+res);
        }
        return res;
    }
    private double binaryLogitsObj(int k) {
        double sum = 0;
        int L = dataSet.getNumClasses();
        LogisticRegression[] logisticRegressions = bmmClassifier.binaryLogitRegressions[k];
        for (int l=0; l<L; l++) {
            WeightedLogisticLoss logisticLoss = new WeightedLogisticLoss(logisticRegressions[l],
                    dataSet, gammasT[k], targetsDistributions[l], gaussianPriorforLogit);
            sum += logisticLoss.getValue();
        }
        return sum;
    }


    public Terminator getTerminator() {
        return terminator;
    }
}
