package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Entropy;
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




        IdTranslator idTranslator = testSet.getIdTranslator();
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(testSet.getRow(i));
            double perplexity = Math.pow(2, Entropy.entropy2Based(proportions));

            if (i==9960){
                System.out.println("----------------------------------------------");
                System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
                System.out.println("labels = "+trueLabel);
                BMMInspector.visualizePrediction(bmmClassifier, testSet.getRow(i));
            }

//            if (trueLabel.getMatchedLabels().size()>=2&&perplexity>1.5&&bmmClassifier.predict(testSet.getRow(i)).equals(trueLabel)){
//                System.out.println("----------------------------------------------");
//                System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
//                System.out.println("labels = "+trueLabel);
//                BMMInspector.visualizePrediction(bmmClassifier,testSet.getRow(i));
//            }
        }
    }
}
