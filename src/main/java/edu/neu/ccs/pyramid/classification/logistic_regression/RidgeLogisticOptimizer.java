package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.*;

/**
 * Created by chengli on 10/7/15.
 */
public class RidgeLogisticOptimizer {
    private Optimizer optimizer;
    private Optimizable.ByGradientValue function;



    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[][] targetDistributions, double gaussianPriorVariance) {
        logisticRegression.setFeatureExtraction(false);
        this.function = new KLLogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        this.optimizer = new LBFGS(function);
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, ClfDataSet dataSet, double gaussianPriorVariance) {
        double[][] targetDistributions = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<labels.length;i++){
            int label = labels[i];
            targetDistributions[i][label]=1;
        }
        logisticRegression.setFeatureExtraction(false);
        this.function = new KLLogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        this.optimizer = new LBFGS(function);
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
