//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
//import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
//import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.DataSetUtil;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.feature.FeatureUtility;
//import edu.neu.ccs.pyramid.feature.Ngram;
//import edu.neu.ccs.pyramid.util.SetUtil;
//import org.apache.mahout.math.Vector;
//
//import java.io.File;
//import java.util.Comparator;
//import java.util.HashSet;
//import java.util.List;
//import java.util.Set;
//import java.util.stream.Collectors;
//
///**
//* todo: fix
//* logistic gradient feature selection
//* Created by chengli on 3/20/15.
//*/
//public class Exp73 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("please specify the config file");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//        check(config);
//
//
//
//    }
//
//    private static void check(Config config) throws Exception{
//        String input = config.getString("input.folder");
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(dataSet.getNumClasses());
//        priorProbClassifier.fit(dataSet);
//        if (config.getBoolean("checkOverlap")){
//            for (int k=0;k<dataSet.getNumClasses();k++){
//                System.out.println("features for class "+k);
//                Set<String> set1 = gradientSelection(config, dataSet, k, priorProbClassifier);
//                Set<String> set2 = logisticSelection(config,k);
//                System.out.println("number of overlapped features = "+ SetUtil.intersect(set1,set2).size());
//
//            }
//        }
//
//        dump(config,dataSet,priorProbClassifier);
//
//    }
//
//    private static double utility(ClfDataSet dataSet, int classIndex, int featureIndex, PriorProbClassifier priorProbClassifier){
//        Vector vector = dataSet.getColumn(featureIndex);
//        int[] labels = dataSet.getLabels();
//        //actual and predicted
//        double[] counts = new double[2];
//        for (Vector.Element element: vector.nonZeroes()){
//            int dataPoint = element.index();
//            counts[1] += 1;
//            if (labels[dataPoint]==classIndex){
//                counts[0] += 1;
//            }
//        }
//        return counts[0] - counts[1]*priorProbClassifier.getClassProbs()[classIndex];
//    }
//
//    protected static Set<String> gradientSelection(Config config, ClfDataSet dataSet, int classIndex, PriorProbClassifier priorProbClassifier){
//        int limit = config.getInt("limit");
//        List<FeatureUtility> top = dataSet.getFeatureList().getAll().parallelStream().map(feature -> {
//            int index = feature.getIndex();
//            double quality = utility(dataSet, classIndex, index, priorProbClassifier);
//            FeatureUtility featureUtility = new FeatureUtility(feature);
//            featureUtility.setUtility(quality);
//            return featureUtility;
//        }).sorted(Comparator.comparing(FeatureUtility::getUtility).reversed())
//                .limit(limit).collect(Collectors.toList());
//        return top.stream().map(featureUtility -> ((Ngram)featureUtility.getFeature()).getNgram())
//                .collect(Collectors.toSet());
//
//    }
//
//    private static Set<String> logisticSelection(Config config, int classIndex) throws Exception{
//        File model = new File(config.getString("input.model"));
//        int limit = config.getInt("limit");
//        LogisticRegression logisticRegression = LogisticRegression.deserialize(model);
//        return LogisticRegressionInspector.topFeatures(logisticRegression, classIndex, limit)
//                .stream().map(featureUtility ->((Ngram)featureUtility.getFeature()).getNgram())
//                .collect(Collectors.toSet());
//    }
//
//    private static Set<Integer> simpleSelectedIndices(Config config,ClfDataSet dataSet, int classIndex, PriorProbClassifier priorProbClassifier){
//        int limit = config.getInt("limit");
//        List<FeatureUtility> top = dataSet.getFeatureList().getAll().parallelStream().map(feature -> {
//            int index = feature.getIndex();
//            double quality = utility(dataSet, classIndex, index, priorProbClassifier);
//            FeatureUtility featureUtility = new FeatureUtility(feature);
//            featureUtility.setUtility(quality);
//            return featureUtility;
//        }).sorted(Comparator.comparing(FeatureUtility::getUtility).reversed())
//                .limit(limit).collect(Collectors.toList());
//        return top.stream().map(featureUtility -> featureUtility.getFeature().getIndex())
//                .collect(Collectors.toSet());
//    }
//
//
//    private static List<Integer> simpleSelectedIndices(Config config,ClfDataSet dataSet, PriorProbClassifier priorProbClassifier) {
//        Set<Integer> integers = new HashSet<>();
//        for (int k = 0; k < dataSet.getNumClasses(); k++) {
//            integers.addAll(simpleSelectedIndices(config, dataSet, k, priorProbClassifier));
//        }
//
//        return integers.stream().sorted().collect(Collectors.toList());
//    }
//
//    private static void dump(Config config,ClfDataSet dataSet, PriorProbClassifier priorProbClassifier)throws Exception{
//        List<Integer> list = simpleSelectedIndices(config,dataSet,priorProbClassifier);
//        ClfDataSet trimed = DataSetUtil.sampleFeatures(dataSet, list);
//        TRECFormat.save(trimed,new File(config.getString("output.folder"),"train.trec"));
//        String input = config.getString("input.folder");
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input, "test.trec"),
//                DataSetType.CLF_SPARSE, true);
//        ClfDataSet trimedTest = DataSetUtil.sampleFeatures(testSet, list);
//        TRECFormat.save(trimedTest,new File(config.getString("output.folder"),"test.trec"));
//    }
//}
