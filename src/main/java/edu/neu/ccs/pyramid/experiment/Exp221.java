package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
import edu.neu.ccs.pyramid.util.Serialization;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * visualize prediction
 * Created by chengli on 1/14/16.
 */
public class Exp221 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);
        MultiLabelClfDataSet trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SPARSE, true);


        List<MultiLabel> list = DataSetUtil.gatherMultiLabels(trainSet);
        Set<MultiLabel> set = new HashSet<>();
        set.addAll(list);


        BMMClassifier bmmClassifier = (BMMClassifier) Serialization.deserialize(config.getString("input.model"));

        IdTranslator idTranslator = testSet.getIdTranslator();
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            if (!set.contains(trueLabel)){
                if (bmmClassifier.predict(testSet.getRow(i)).equals(trueLabel)){
                    System.out.println("----------------------------------------------");
                    System.out.println("data point "+i+", extId="+idTranslator.toExtId(i));
                    System.out.println("labels = "+trueLabel);
                    BMMInspector.visualizePrediction(bmmClassifier,testSet.getRow(i));
                }
            }
        }
    }
}
