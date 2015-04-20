package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.util.Grid;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * elastic-net logistic regression
 * Created by chengli on 2/25/15.
 */
public class Exp70 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }


    }


    public static void test(Config config) throws Exception{
        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);
        List<Double> l1Ratios = Grid.uniform(config.getDouble("train.l1Ratio.min"),
                config.getDouble("train.l1Ratio.max"), config.getInt("train.l1Ratio.size"))
                .stream().sorted().collect(Collectors.toList());

        List<Double> regularizations = Grid.logUniform(config.getDouble("train.regularization.min"),
                config.getDouble("train.regularization.max"), config.getInt("train.regularization.size"))
                .stream().sorted(comparator.reversed()).collect(Collectors.toList());

        ClfDataSet testSet = loadTest(config);

        List<Performance> performanceList = new ArrayList<>();

        for (double l1ratio: l1Ratios){
            for (double regularization: regularizations){
                File outputFolder = new File(new File(config.getString("output.folder"),""+l1ratio),""+regularization);
                File serFile = new File(outputFolder,"model");
                LogisticRegression logisticRegression = LogisticRegression.deserialize(serFile);
                double acc = Accuracy.accuracy(logisticRegression, testSet);
                Performance performance = new Performance(regularization,l1ratio,acc);
                performanceList.add(performance);
            }
        }


        for (double l1ratio: l1Ratios){
            System.out.println("===================");
            System.out.println("l1Ratio = "+l1ratio);

            Performance best = performanceList.stream().filter(performance -> performance.l1Ratio == l1ratio)
                    .max(Comparator.comparing(Performance::getAccuracy)).get();
            double regularization = best.regularization;
            File outputFolder = new File(new File(config.getString("output.folder"),""+l1ratio),""+regularization);
            File serFile = new File(outputFolder,"model");
            LogisticRegression logisticRegression = LogisticRegression.deserialize(serFile);
                    System.out.println("regularization = "+best.regularization);
            System.out.println("accuracy = "+best.accuracy);
            System.out.println("number of used features in each class = "+
                    Arrays.toString(LogisticRegressionInspector.numOfUsedFeaturesEachClass(logisticRegression)));

            System.out.println("number of used features in all classes = "+
                    LogisticRegressionInspector.numOfUsedFeaturesCombined(logisticRegression));

            System.out.println(LogisticRegressionInspector.checkNgramUsage(logisticRegression));
        }

        System.out.println("**********************");
        System.out.println("overall best one");
        Performance best = performanceList.stream()
                .max(Comparator.comparing(Performance::getAccuracy)).get();
        double regularization = best.regularization;
        double l1ratio = best.l1Ratio;
        File outputFolder = new File(new File(config.getString("output.folder"),""+l1ratio),""+regularization);
        File serFile = new File(outputFolder,"model");
        LogisticRegression logisticRegression = LogisticRegression.deserialize(serFile);
        System.out.println("regularization = "+best.regularization);
        System.out.println("l1Ratio = "+best.l1Ratio);
        System.out.println("accuracy = "+best.accuracy);
        System.out.println("number of used features in each class = "+
                Arrays.toString(LogisticRegressionInspector.numOfUsedFeaturesEachClass(logisticRegression)));

        System.out.println("number of used features in all classes = "+
                LogisticRegressionInspector.numOfUsedFeaturesCombined(logisticRegression));
        System.out.println(LogisticRegressionInspector.checkNgramUsage(logisticRegression));

        File bestModelFolder = new File(config.getString("output.folder"),"best");

        bestModelFolder.mkdirs();
        FileUtils.copyFileToDirectory(serFile,bestModelFolder);

    }


    public static void train(Config config) throws Exception{

        ClfDataSet dataSet = loadTrain(config);


        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);

        List<Double> l1Ratios = Grid.uniform(config.getDouble("train.l1Ratio.min"),
                config.getDouble("train.l1Ratio.max"), config.getInt("train.l1Ratio.size"))
                .stream().sorted().collect(Collectors.toList());

        List<Double> regularizations = Grid.logUniform(config.getDouble("train.regularization.min"),
                config.getDouble("train.regularization.max") , config.getInt("train.regularization.size"))
                .stream().sorted(comparator.reversed()).collect(Collectors.toList());



        for (double l1ratio: l1Ratios){

            LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
            for (double regularization: regularizations){
                ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                        .setEpsilon(0.01).setL1Ratio(l1ratio).setRegularization(regularization).build();

                System.out.println("=================================");
                System.out.println("l1Ratio= "+l1ratio );
                System.out.println("regularization = "+regularization);
                StopWatch stopWatch = new StopWatch();
                stopWatch.start();
                trainer.train();
                System.out.println("time spent = " + stopWatch);
                System.out.println("number of used features in each class = "+ Arrays.toString(LogisticRegressionInspector.numOfUsedFeaturesEachClass(logisticRegression)));

                File outputFolder = new File(new File(config.getString("output.folder"),""+l1ratio),""+regularization);
                outputFolder.mkdirs();
                File serFile = new File(outputFolder,"model");
                logisticRegression.serialize(serFile);




            }
        }
    }

    private static ClfDataSet loadTrain(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);

        ClfDataSet subSet = null;
        if (config.getBoolean("featureSampling")){
            if (config.getBoolean("featureSampling.byField")){
                List<Integer> indices = sampleFeatures(config);
                subSet = DataSetUtil.sampleFeatures(dataSet, indices);
            } else if (config.getBoolean("featureSampling.byNumber")){
                int numFeatures = config.getInt("featureSampling.numFeaturesToUse");
                subSet = DataSetUtil.sampleFeatures(dataSet, numFeatures);
            }
        } else {
            subSet = dataSet;
        }

        System.out.println(subSet.getMetaInfo());
        return subSet;
    }


    private static ClfDataSet loadTest(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.testData")),
                DataSetType.CLF_SPARSE, true);

        ClfDataSet subSet = null;
        if (config.getBoolean("featureSampling")){
            if (config.getBoolean("featureSampling.byField")){
                List<Integer> indices = sampleFeatures(config);
                subSet = DataSetUtil.sampleFeatures(dataSet, indices);
            } else if (config.getBoolean("featureSampling.byNumber")){
                int numFeatures = config.getInt("featureSampling.numFeaturesToUse");
                subSet = DataSetUtil.sampleFeatures(dataSet, numFeatures);
            }
        } else {
            subSet = dataSet;
        }

        System.out.println(subSet.getMetaInfo());
        return subSet;
    }

    private static List<Integer> sampleFeatures(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);
        Set<String> fields = new HashSet<>(config.getStrings("featureSampling.fields"));
        Set<String> fieldPrefixes = new HashSet<>(config.getStrings("featureSampling.fieldPrefixes"));

        FeatureList featureList = dataSet.getFeatureList();
        List<Integer> collect = featureList.getAll().stream().filter(feature -> {
                    boolean cond1 = !(feature instanceof Ngram);
                    boolean cond2 = (feature instanceof Ngram) && fields.contains(((Ngram) feature).getField());
                    boolean cond3 = (feature instanceof Ngram) && matchPrefix((((Ngram) feature).getField()), fieldPrefixes);
                    return cond1 || cond2 || cond3;
                }
        ).map(feature -> feature.getIndex()).collect(Collectors.toList());
        return collect;
    }

    private static boolean matchPrefix(String field, Set<String> prefixes){
        for (String prefix: prefixes){
            if (field.startsWith(prefix)){
                return true;
            }
        }
        return false;
    }


    private static class Performance{
        private double regularization;
        private double l1Ratio;
        private double accuracy;

        public Performance(double regularization, double l1Ratio, double accuracy) {
            this.regularization = regularization;
            this.l1Ratio = l1Ratio;
            this.accuracy = accuracy;
        }

        public double getAccuracy() {
            return accuracy;
        }
    }
}
