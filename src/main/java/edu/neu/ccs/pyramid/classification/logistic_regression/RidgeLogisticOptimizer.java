package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.ConjugateGradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;

/**
 * Created by chengli on 10/7/15.
 */
public class RidgeLogisticOptimizer {
    private Optimizer optimizer;

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public RidgeLogisticOptimizer(LogisticRegression logisticRegression, DataSet dataSet,
                                  double[][] targetDistributions, double gaussianPriorVariance) {
        logisticRegression.setFeatureExtraction(false);
        KLLogisticLoss function = new KLLogisticLoss(logisticRegression,dataSet,
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
        KLLogisticLoss function = new KLLogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        this.optimizer = new LBFGS(function);
    }

    public void optimize(){
        this.optimizer.optimize();
    }




}
