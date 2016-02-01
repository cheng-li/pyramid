package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.Serialization;

/**
 *
 * Created by chengli on 1/31/16.
 */
public class Exp146 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);
        MultiLabelClassifier classifier = (MultiLabelClassifier)Serialization.deserialize(config.getString("input.model"));
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabel[] prediction = classifier.predict(testSet);
        System.out.println("accuracy = "+ Accuracy.accuracy(testSet.getMultiLabels(),prediction));
        System.out.println("overlap = "+ Overlap.overlap(testSet.getMultiLabels(),prediction));
        System.out.println("hamming loss = "+ HammingLoss.hammingLoss(testSet.getMultiLabels(),prediction,testSet.getNumClasses()));
    }
}
