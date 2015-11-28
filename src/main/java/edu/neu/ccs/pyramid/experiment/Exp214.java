package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import java.io.File;

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
        show(config,bmm);

    }

    private static void show(Config config, BMMClassifier bmmClassifier) throws Exception{
        int numClusters = bmmClassifier.getNumClusters();
        int numClasses = bmmClassifier.getNumClasses();
        String output = config.getString("output.folder");
        new File(output).mkdirs();
        Classifier.ProbabilityEstimator[][] classifiers = bmmClassifier.getBinaryClassifiers();
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numClasses;l++){
                LogisticRegression logisticRegression = (LogisticRegression)classifiers[k][l];
                Vector vector = logisticRegression.getWeights().getWeightsForClass(1);
                StringBuilder sb = new StringBuilder();
                for (Vector.Element element: vector.all()){
                    sb.append(element.index()).append(": ").append(element.get()).append("\n");
                }
                double bias = vector.get(0);
                boolean interesting = (bias>-10)&&(bias<10);
                File file = new File(output,interesting+""+k+"_"+l);
                FileUtils.writeStringToFile(file,sb.toString());
            }
        }
    }

    private static BMMClassifier loadModel(Config config) throws Exception{
        String model = config.getString("model");
        BMMClassifier bmm = (BMMClassifier)Serialization.deserialize(model);
        return bmm;
    }


}
