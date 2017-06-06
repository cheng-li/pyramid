package edu.neu.ccs.pyramid.core.multilabel_classification;

import edu.neu.ccs.pyramid.core.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.core.classification.logistic_regression.RidgeLogisticTrainer;
import edu.neu.ccs.pyramid.core.dataset.*;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * scaling by fitting every class score against every class label match(0/1)
 * Created by chengli on 5/29/15.
 */
public class MLFlatScaling implements MultiLabelClassifier.ClassProbEstimator{

    private static final long serialVersionUID = 1L;
    private MultiLabelClassifier.ClassScoreEstimator scoreEstimator;
    private LogisticRegression logisticRegression;

    public MLFlatScaling(MultiLabelClfDataSet dataSet, MultiLabelClassifier.ClassScoreEstimator scoreEstimator) {
        this.scoreEstimator = scoreEstimator;
        int numDataPoints = dataSet.getNumDataPoints();
        int numClasses = dataSet.getNumClasses();
        ClfDataSet clfDataSet = ClfDataSetBuilder.getBuilder().numDataPoints(numDataPoints*numClasses)
                .numFeatures(1).numClasses(2).dense(true)
                .missingValue(false)
                .build();
        int rowIndex = 0;
        for (int i=0;i<numDataPoints;i++){
            double[] scores = scoreEstimator.predictClassScores(dataSet.getRow(i));
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int k=0;k<numClasses;k++){
                clfDataSet.setFeatureValue(rowIndex,0,scores[k]);
                if (multiLabel.matchClass(k)){
                    clfDataSet.setLabel(rowIndex,1);
                }
                rowIndex += 1;
            }
        }

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setEpsilon(1)
                .setGaussianPriorVariance(100)
                .setHistory(5)
                .build();

        logisticRegression = trainer.train(clfDataSet);
    }


    @Override
    public double[] predictClassProbs(Vector vector) {
        double[] scores = scoreEstimator.predictClassScores(vector);
        double[] probs = new double[scores.length];
        for (int k=0;k<scores.length;k++){
            Vector scoreFeatureVector = new DenseVector(1);
            scoreFeatureVector.set(0,scores[k]);
            probs[k] = logisticRegression.predictClassProb(scoreFeatureVector,1);
        }
        return probs;
    }

    @Override
    public int getNumClasses() {
        return scoreEstimator.getNumClasses();
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
}
