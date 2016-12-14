package edu.neu.ccs.pyramid.multilabel_classification;


import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticRegression;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticTrainer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * multi-label across classes platt scaling
 * Created by chengli on 4/19/15.
 */
public class MLACPlattScaling implements MultiLabelClassifier.ClassProbEstimator,
        MultiLabelClassifier.AssignmentProbEstimator{
    private static final long serialVersionUID = 1L;
    private MultiLabelClassifier.ClassScoreEstimator scoreEstimator;
    private MLLogisticRegression logisticRegression;


    public MLACPlattScaling(MultiLabelClfDataSet dataSet, MultiLabelClassifier.ClassScoreEstimator scoreEstimator) {
        this.scoreEstimator = scoreEstimator;
        MultiLabelClfDataSet scoreDataSet = MLClfDataSetBuilder.getBuilder().numDataPoints(dataSet.getNumDataPoints())
                .numFeatures(dataSet.getNumClasses()).numClasses(dataSet.getNumClasses()).dense(true)
                .missingValue(false).build();
        for (int i=0;i<scoreDataSet.getNumDataPoints();i++){
            scoreDataSet.addLabels(i,dataSet.getMultiLabels()[i].getMatchedLabels());
        }

        for (int i=0;i<scoreDataSet.getNumDataPoints();i++){
            double[] scores = scoreEstimator.predictClassScores(dataSet.getRow(i));
            for (int k=0;k<scoreDataSet.getNumClasses();k++){
                scoreDataSet.setFeatureValue(i,k,scores[k]);
            }
        }

        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(100000)
                .build();
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(scoreDataSet);
        this.logisticRegression = trainer.train(scoreDataSet,assignments);
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        double[] scores = scoreEstimator.predictClassScores(vector);
        Vector scoreVector = new DenseVector(scores.length);
        for (int i=0;i<scores.length;i++){
            scoreVector.set(i,scores[i]);
        }
        return this.logisticRegression.predictClassProbs(scoreVector);
    }

    @Override
    public int getNumClasses() {
        return logisticRegression.getNumClasses();
    }

    @Override
    public MultiLabel predict(Vector vector) {
        return scoreEstimator.predict(vector);
    }

    @Override
    public FeatureList getFeatureList() {
        return logisticRegression.getFeatureList();
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return logisticRegression.getLabelTranslator();
    }

    @Override
    public double predictLogAssignmentProb(Vector vector, MultiLabel assignment) {
        //todo
        return 0;
    }

    @Override
    public double predictAssignmentProb(Vector vector, MultiLabel assignment) {
        double[] scores = scoreEstimator.predictClassScores(vector);
        Vector scoreVector = new DenseVector(scores.length);
        for (int i=0;i<scores.length;i++){
            scoreVector.set(i,scores[i]);
        }
        return logisticRegression.predictAssignmentProb(scoreVector,assignment);
    }
}
