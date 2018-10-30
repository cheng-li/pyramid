package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

public class PlattScaling implements Serializable {
    private static final long serialVersionUID = 1L;
    private LogisticRegression logisticRegression;


    public PlattScaling(double[] scores, int[] labels) {
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numClasses(2).numDataPoints(scores.length).numFeatures(1)
                .dense(true).missingValue(false).build();
        for (int i=0;i<scores.length;i++){
            dataSet.setFeatureValue(i,0,scores[i]);
            dataSet.setLabel(i,labels[i]);
        }

        this.logisticRegression = new LogisticRegression(2,dataSet.getNumFeatures());
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticRegression,dataSet,1000000,true);
        ridgeLogisticOptimizer.optimize();
    }

    /**
     *
     * @param scores
     * @param positiveProbs target positive label probability
     */
    public PlattScaling(double[] scores, double[] positiveProbs) {
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(scores.length).numFeatures(1)
                .dense(true).missingValue(false).build();
        double[][] targetDis = new double[positiveProbs.length][2];
        for (int i=0;i<scores.length;i++){
            dataSet.setFeatureValue(i,0,scores[i]);
            targetDis[i][1] = positiveProbs[i];
            targetDis[i][0] = 1-positiveProbs[i];
        }

        this.logisticRegression = new LogisticRegression(2,dataSet.getNumFeatures());
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(logisticRegression,dataSet,targetDis,1000000,true);
        ridgeLogisticOptimizer.optimize();
    }

    public double transform(double uncalibrated){
        Vector vector = new DenseVector(1);
        vector.set(0, uncalibrated);
        return logisticRegression.predictClassProb(vector,1);
    }
}
