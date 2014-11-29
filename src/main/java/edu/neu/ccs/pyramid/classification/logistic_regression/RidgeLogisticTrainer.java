package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


/**
 * Created by chengli on 11/28/14.
 */
public class RidgeLogisticTrainer {
    public static LogisticRegression train(ClfDataSet clfDataSet, double regularization){
        Vector C = new DenseVector(clfDataSet.getNumDataPoints());
        for (int i=0;i<C.size();i++){
            C.set(i,regularization);
        }
        RidgeLogisticLoss function = new RidgeLogisticLoss(clfDataSet, C);
        LogisticRegression logisticRegression = new LogisticRegression(clfDataSet.getNumFeatures());
        Tron tron = new Tron(function,0.000001);
        tron.tron(logisticRegression.weights);
        return logisticRegression;
    }
}
