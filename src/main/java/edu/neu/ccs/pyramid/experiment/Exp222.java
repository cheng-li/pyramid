package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * local version of exp221
 * Created by chengli on 1/16/16.
 */
public class Exp222 {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{

        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "meka_imdb/1/data_sets/test"),
                DataSetType.ML_CLF_SPARSE, true);
        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize(new File(TMP,"model"));


        BMMClassifier independent = (BMMClassifier)Serialization.deserialize(new File(TMP,"inde"));

        MultiLabel[] indePredictions = independent.predict(testSet);
        MultiLabel[] bmmPredictions = bmmClassifier.predict(testSet);

        boolean measure = false;

        if (measure){
            System.out.println("independent measures");

            LabelBasedMeasures indeMeasure = new LabelBasedMeasures(testSet,indePredictions);
            System.out.println(indeMeasure);
            MacroMeasures indemacro = new MacroMeasures(testSet,indePredictions);
            System.out.println(indemacro);
            System.out.println("acc = "+ Accuracy.accuracy(testSet.getMultiLabels(),indePredictions));
            System.out.printf("overlap = "+ Overlap.overlap(testSet.getMultiLabels(),indePredictions));

            System.out.println("bmm measures");

            LabelBasedMeasures bmmMeasures = new LabelBasedMeasures(testSet,bmmPredictions);
            System.out.println(bmmMeasures);
            MacroMeasures bmmmacro = new MacroMeasures(testSet,bmmPredictions);
            System.out.println(bmmmacro);
            System.out.println("acc = "+ Accuracy.accuracy(testSet.getMultiLabels(),bmmPredictions));
            System.out.println("overlap = " + Overlap.overlap(testSet.getMultiLabels(), bmmPredictions));
        }






        IdTranslator idTranslator = testSet.getIdTranslator();
        LabelTranslator labelTranslator = testSet.getLabelTranslator();
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(testSet.getRow(i));
            double perplexity = Math.pow(2, Entropy.entropy2Based(proportions));
            MultiLabel pred = bmmClassifier.predict(testSet.getRow(i));
            MultiLabel expectation = bmmClassifier.predictByExpectation(testSet.getRow(i));
            MultiLabel indePred = independent.predict(testSet.getRow(i));

//            boolean condition = trueLabel.getMatchedLabels().size()>=2&&perplexity>1.5&&pred.equals(trueLabel)&&!indePred.equals(trueLabel)&&!expectation.equals(trueLabel);
            boolean condition = i==526;

            if (condition){
                System.out.println("----------------------------------------------");
                System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
                System.out.println("labels = "+trueLabel.toStringWithExtLabels(labelTranslator));
                System.out.println("independent prediction = "+indePred.toStringWithExtLabels(labelTranslator));
                System.out.println("bmm probability for independent prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),indePred));
                System.out.println("prediction = "+pred.toStringWithExtLabels(labelTranslator));
                System.out.println("bmm probability for bmm prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),pred));
                System.out.println("expectation = "+expectation.toStringWithExtLabels(labelTranslator));
                System.out.println("bmm probability for expectation prediction = "+bmmClassifier.predictAssignmentProb(testSet.getRow(i),expectation));
                BMMInspector.covariance(bmmClassifier,testSet.getRow(i),labelTranslator);
                BMMInspector.visualizePrediction(bmmClassifier,testSet.getRow(i));

                for (int label: indePredictions[i].getMatchedLabels()){
                    LogisticRegression logisticRegression = (LogisticRegression)independent.getBinaryClassifiers()[0][label];
                    logisticRegression.setFeatureList(testSet.getFeatureList());
                    ClassScoreCalculation classScoreCalculation = LogisticRegressionInspector.decisionProcess(logisticRegression, testSet.getLabelTranslator(), testSet.getRow(i), 1, 10);
                    System.out.println("for class "+label);
                    System.out.println(classScoreCalculation);
                }
            }
        }
    }
}
