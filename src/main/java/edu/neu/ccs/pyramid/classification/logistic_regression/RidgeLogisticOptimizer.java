package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.*;

/**
 * Created by chengli on 10/7/15.
 */
public class RidgeLogisticOptimizer{
    private Optimizer optimizer;
    private Optimizable.ByGradientValue function;
    private boolean isParallel = false;


    public RidgeLogisticOptimizer(LogisticLoss logisticLoss){
        this.function = logisticLoss;
        this.optimizer = new LBFGS(function);
        this.optimizer.getTerminator().setAbsoluteEpsilon(0.1);
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  int[] labels, double gaussianPriorVariance, boolean parallel) {
        this(logisticRegression,dataSet,labelsToDistributions(labels,logisticRegression.getNumClasses()),gaussianPriorVariance, parallel);
    }

    /**
     *
     * @param logisticRegression
     * @param dataSet
     * @param targetDistributions [# data points][# labels]
     * @param gaussianPriorVariance
     */
    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[][] targetDistributions, double gaussianPriorVariance,
                                  boolean parallel) {
        this.function = new LogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance, parallel);
        this.optimizer = new LBFGS(function);
        this.optimizer.getTerminator().setAbsoluteEpsilon(0.1);
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, ClfDataSet dataSet, double gaussianPriorVariance, boolean parallel) {

        this.function = new LogisticLoss(logisticRegression,dataSet, gaussianPriorVariance, parallel);
        this.optimizer = new LBFGS(function);
        this.optimizer.getTerminator().setAbsoluteEpsilon(0.1);
    }

    /**
     *
     * @param logisticRegression
     * @param dataSet
     * @param weights
     * @param targetsDistribution [# data points][# labels]
     * @param gaussianPriorVar
     */
    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[] weights, double[][] targetsDistribution, double gaussianPriorVar, boolean parallel) {
        this.function = new LogisticLoss(logisticRegression, dataSet, weights, targetsDistribution, gaussianPriorVar, parallel);
        this.optimizer = new LBFGS(function);
        this.optimizer.getTerminator().setAbsoluteEpsilon(0.1);
    }


    public void optimize(){
        this.optimizer.optimize();
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public Optimizable.ByGradientValue getFunction() {
        return function;
    }

    private static double[][] labelsToDistributions(int[] labels, int numClass){
        int numData = labels.length;
        double[][] distribution = new double[numData][numClass];
        for (int i=0;i<numData;i++){
            int label = labels[i];
            distribution[i][label] = 1;
        }
        return distribution;
    }

}
