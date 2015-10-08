package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.LBFGS;

/**
 * Created by chengli on 10/7/15.
 */
public class RidgeLogisticOptimizer {
    private double gaussianPriorVariance = 1;

    public RidgeLogisticOptimizer(double gaussianPriorVariance) {
        this.gaussianPriorVariance = gaussianPriorVariance;
    }

    public void optimize(LogisticRegression logisticRegression, DataSet dataSet,
                         double[][] targetDistributions){
        logisticRegression.setFeatureExtraction(false);
        KLLogisticLoss function = new KLLogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.optimize();
    }

    public void optimize(LogisticRegression logisticRegression, ClfDataSet dataSet){
        double[][] targetDistributions = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<labels.length;i++){
            int label = labels[i];
            targetDistributions[i][label]=1;
        }
        logisticRegression.setFeatureExtraction(false);
        KLLogisticLoss function = new KLLogisticLoss(logisticRegression,dataSet,
                targetDistributions,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.optimize();
    }



}
