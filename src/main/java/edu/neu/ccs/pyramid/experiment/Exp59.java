package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.FeatureUtility;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * check whether important featureList in one dataset appear in another dataset
 * Created by chengli on 1/24/15.
 */
public class Exp59 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        List<List<String>> goldenFeatures = loadGoldenFeatures(config);
        Set<String> realFeatures = loadRealFeatures(config);
        int numClasses = goldenFeatures.size();
        for (int k=0;k<numClasses;k++){
            List<String> featuresK = goldenFeatures.get(k);
            System.out.println("for class "+k+", missing important featureList:");
            System.out.println();
            System.out.println(featuresK.stream().filter(feature -> !realFeatures.contains(feature)).collect(Collectors.toList()));
        }



    }

    static List<List<String>> loadGoldenFeatures(Config config) throws Exception{
        List<List<String>> features = new ArrayList<>();
        File modelFile = new File(config.getString("input.golden.folder"),"model");
        LogisticRegression logisticRegression = LogisticRegression.deserialize(modelFile);
        int limit = config.getInt("topFeature.limit");
        for (int k=0;k<logisticRegression.getNumClasses();k++) {
            features.add(new ArrayList<>());
            features.get(k).addAll(LogisticRegressionInspector.topFeatures(logisticRegression, k, limit)
                    .stream().map(FeatureUtility::getName).collect(Collectors.toList()));
        }
        return features;
    }

    static Set<String> loadRealFeatures(Config config) throws Exception{
        File data = new File(config.getString("input.real.folder"),"test.trec");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(data, DataSetType.CLF_SPARSE, true);
        Set<String> set1 = new HashSet<>();
        IntStream.range(0, dataSet1.getNumFeatures()).mapToObj(i-> dataSet1.getFeatureSetting(i).getFeatureName())
                .forEach(set1::add);
        return set1;
    }
}
