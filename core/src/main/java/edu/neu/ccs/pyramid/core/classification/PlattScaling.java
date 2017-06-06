package edu.neu.ccs.pyramid.core.classification;

import edu.neu.ccs.pyramid.core.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.core.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/3/15.
 */
public class PlattScaling implements Classifier.ProbabilityEstimator{

    private List<LogisticRegression> logisticRegressions;
    private Classifier.ScoreEstimator scoreEstimator;

    public PlattScaling(ClfDataSet dataSet, Classifier.ScoreEstimator scoreEstimator) {
        this.scoreEstimator = scoreEstimator;
        this.logisticRegressions = new ArrayList<>();
        for (int classIndex = 0; classIndex<dataSet.getNumClasses();classIndex++){
            logisticRegressions.add(fitClassK(dataSet,scoreEstimator,classIndex));
        }
    }


    private static LogisticRegression fitClassK(ClfDataSet dataSet, Classifier.ScoreEstimator scoreEstimator, int classIndex){
        int numDataPoints = dataSet.getNumDataPoints();
        double[] scores = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i -> scoreEstimator.predictClassScore(dataSet.getRow(i),classIndex)).toArray();
        int[] labels = IntStream.range(0,numDataPoints).parallel()
                .map(i -> {
                    if (dataSet.getLabels()[i]==classIndex){
                        return 1;
                    } else {
                        return 0;
                    }
                }).toArray();
        return fitClassK(scores,labels);
    }


    private static LogisticRegression fitClassK(double[] scores, int[] labels){
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numClasses(2).numDataPoints(scores.length).numFeatures(1)
                .dense(true).missingValue(false).build();
        for (int i=0;i<scores.length;i++){
            dataSet.setFeatureValue(i,0,scores[i]);
            dataSet.setLabel(i,labels[i]);
        }

        LogisticRegression logisticRegression = new LogisticRegression(2,dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setRegularization(1.0E-9).setL1Ratio(0).build();
        trainer.optimize();
        return logisticRegression;
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        double[] probs = new double[scoreEstimator.getNumClasses()];
        for (int k=0;k<probs.length;k++){
            probs[k] = predictClassProb(vector,k);
        }
        return probs;
    }

    @Override
    public double predictClassProb(Vector vector, int classIndex) {
        double score = scoreEstimator.predictClassScore(vector,classIndex);
        Vector scoreVector = new DenseVector(1);
        scoreVector.set(0,score);
        return logisticRegressions.get(classIndex).predictClassProb(scoreVector,1);
    }

    @Override
    public int getNumClasses() {
        return scoreEstimator.getNumClasses();
    }

    @Override
    public FeatureList getFeatureList() {
        return scoreEstimator.getFeatureList();
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return scoreEstimator.getLabelTranslator();
    }
}
