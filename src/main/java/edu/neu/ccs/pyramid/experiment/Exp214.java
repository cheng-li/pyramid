package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Serialization;

/**
 * check bmm classifier
 * Created by chengli on 11/24/15.
 */
public class Exp214 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        BMMClassifier bmm = loadModel(config);
        show(bmm);

    }

    private static void show(BMMClassifier bmmClassifier){
        int numClusters = bmmClassifier.getNumClusters();
        int numClasses = bmmClassifier.getNumClasses();
        Classifier.ProbabilityEstimator[][] classifiers = bmmClassifier.getBinaryClassifiers();
        LogisticRegression logisticRegression1 = (LogisticRegression)classifiers[0][0];
        LogisticRegression logisticRegression2 = (LogisticRegression)classifiers[1][0];
        System.out.println("lr in cluster"+0+" ="+logisticRegression1.getWeights().getWeightsForClass(1));
        System.out.println("lr in cluster"+1+" ="+logisticRegression2.getWeights().getWeightsForClass(1));

    }

    private static BMMClassifier loadModel(Config config) throws Exception{
        String model = config.getString("model");
        BMMClassifier bmm = (BMMClassifier)Serialization.deserialize(model);
        return bmm;
    }


}
