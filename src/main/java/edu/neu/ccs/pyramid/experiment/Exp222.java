package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.eval.LabelBasedMeasures;
import edu.neu.ccs.pyramid.eval.MacroMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
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

        System.out.println("independent measures");
        MultiLabel[] indePredictions = independent.predict(testSet);
        LabelBasedMeasures indeMeasure = new LabelBasedMeasures(testSet,indePredictions);
        System.out.println(indeMeasure);
        MacroMeasures indemacro = new MacroMeasures(testSet,indePredictions);
        System.out.println(indemacro);

        System.out.println("bmm measures");
        MultiLabel[] bmmPredictions = bmmClassifier.predict(testSet);
        LabelBasedMeasures bmmMeasures = new LabelBasedMeasures(testSet,bmmPredictions);
        System.out.println(bmmMeasures);
        MacroMeasures bmmmacro = new MacroMeasures(testSet,bmmPredictions);
        System.out.println(bmmmacro);



        IdTranslator idTranslator = testSet.getIdTranslator();
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(testSet.getRow(i));
            double perplexity = Math.pow(2, Entropy.entropy2Based(proportions));
            MultiLabel pred = bmmClassifier.predict(testSet.getRow(i));
            MultiLabel expectation = bmmClassifier.predictByExpectation(testSet.getRow(i));
            MultiLabel indePred = independent.predict(testSet.getRow(i));

            if (trueLabel.getMatchedLabels().size()>=2&&perplexity>1.5&&pred.equals(trueLabel)&&!indePred.equals(trueLabel)&&!expectation.equals(trueLabel)){
                System.out.println("----------------------------------------------");
                System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
                System.out.println("labels = "+trueLabel);
                System.out.println("independent prediction = "+indePred);
                BMMInspector.visualizePrediction(bmmClassifier,testSet.getRow(i));
                System.out.println("prediction = "+pred);
                System.out.println("expectation = "+expectation);
                BMMInspector.covariance(bmmClassifier,testSet.getRow(i));
            }
        }
    }
}
