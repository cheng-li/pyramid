package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * platt scaling for imlgb at the set level
 */
public class IMLGBScaling{
    private LogisticRegression logisticRegression;
    IMLGradientBoosting boosting;

    public IMLGBScaling(IMLGradientBoosting boosting, MultiLabelClfDataSet multiLabelClfDataSet) {
        System.out.println("calibrating");
        this.boosting = boosting;
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numClasses(2).numDataPoints(multiLabelClfDataSet.getNumDataPoints()).numFeatures(1)
                .dense(true).missingValue(false).build();
        SubsetAccPredictor predictor = new SubsetAccPredictor(boosting);
        IntStream.range(0, multiLabelClfDataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    MultiLabel pre = predictor.predict(multiLabelClfDataSet.getRow(i));
                    double score = boosting.predictAssignmentScore(multiLabelClfDataSet.getRow(i),pre);
                    dataSet.setFeatureValue(i, 0, score);
                    if (pre.equals(multiLabelClfDataSet.getMultiLabels()[i])){
                        dataSet.setLabel(i,1);
                    } else {
                        dataSet.setLabel(i, 0);
                    }
                });
        this.logisticRegression = new LogisticRegression(2,dataSet.getNumFeatures());
        RidgeLogisticOptimizer logisticOptimizer = new RidgeLogisticOptimizer(logisticRegression, dataSet,1000000, true);
        logisticOptimizer.optimize();
        System.out.println("calibration done");
    }

    public double calibratedProb(Vector vector, MultiLabel multiLabel){
        double score = boosting.predictAssignmentScore(vector, multiLabel);
        Vector scoreVector = new DenseVector(1);
        scoreVector.set(0, score);
        return logisticRegression.predictClassProbs(scoreVector)[1];
    }
}
