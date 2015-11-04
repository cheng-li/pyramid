package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.*;

/**
 * Created by chengli on 10/7/15.
 */
public class RidgeLogisticOptimizer implements Parallelizable{
    private Optimizer optimizer;
    private Optimizable.ByGradientValue function;
    private boolean isParallel = false;


    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[][] targetDistributions, double gaussianPriorVariance) {
        this.function = new LogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        this.optimizer = new LBFGS(function);
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, ClfDataSet dataSet, double gaussianPriorVariance) {

        this.function = new LogisticLoss(logisticRegression,dataSet, gaussianPriorVariance);
        this.optimizer = new LBFGS(function);
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[] weights, double[][] targetsDistribution, double gaussianPriorVar) {
        logisticRegression.setFeatureExtraction(false);
        this.function = new LogisticLoss(logisticRegression, dataSet, weights, targetsDistribution, gaussianPriorVar);
        this.optimizer = new LBFGS(function);
    }

    @Override
    public void setParallelism(boolean isParallel) {
        this.isParallel = isParallel;
        function.setParallelism(this.isParallel());
    }

    @Override
    public boolean isParallel() {
        return this.isParallel;
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

}
