package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;

import java.util.stream.IntStream;

public class IMLGBInspection {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        IMLGradientBoosting boosting = (IMLGradientBoosting) Serialization.deserialize(config.getString("model"));
        System.out.println("average number of features selected by each binary classifier = "+IntStream.range(0, boosting.getNumClasses()).mapToDouble(l-> IMLGBInspector.getSelectedFeatures(boosting,l).size()).average().getAsDouble());
        System.out.println("total number of features selected (union) = "+IMLGBInspector.getSelectedFeatures(boosting).size());

    }
}
