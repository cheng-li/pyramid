package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * check distribution of good ngrams
 * Created by chengli on 1/27/15.
 */
public class Exp63 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);



        List<List<FeatureUtility>> goodNgrams  = getGoodNgrams(config);

        checkNgrams(goodNgrams);
        List<List<Integer>> rankedDocs = rankDocs(config);
        Map<Integer, Integer> docToRank = docToRank(rankedDocs);
        Map<String, Integer> featureToIndex = featureToIndex(config);
        ClfDataSet dataSet = loadGoodDataSet(config);
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("==========================");
            System.out.println("for class "+k);
            List<FeatureUtility> features = goodNgrams.get(k);
            for (int i=0;i<features.size();i++){
                String feature = features.get(i).getName();
                System.out.println(showDistribution(i,feature, k, dataSet, featureToIndex, docToRank,config));
            }
        }

        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("==========================");
            System.out.println("for class "+k);
            List<Integer> list = rankedDocs.get(k);
            for (int i=0;i<list.size();i++){
                System.out.println(getMatchedGoodFeatures(config, i,list.get(i),goodNgrams,dataSet));
            }
        }

    }


    public static String getMatchedGoodFeatures(Config config, int docRank, int docId, List<List<FeatureUtility>> goodNgrams, ClfDataSet dataSet){


        StringBuilder sb = new StringBuilder();
        Comparator<Pair<String, Integer>> comparator = Comparator.comparing(Pair::getSecond);
        sb.append("docId = ").append(dataSet.getDataPointSetting(docId).getExtId()).append(", ");
        sb.append("doc rank = ").append(docRank).append(", ");
       IntStream.range(0,goodNgrams.size()).forEach(k->
       {
          List<FeatureUtility> matched = goodNgrams.get(k).stream().filter(featureUtility -> dataSet.getRow(docId).get(featureUtility.getIndex())>0)
                  .collect(Collectors.toList());

           sb.append("number of matched features for class ").append(k).append(" = ").append(matched.size()).append(", ");
           sb.append("sum of weights for matched features = ").append(matched.stream()
                   .mapToDouble(FeatureUtility::getUtility).sum()).append(", ");
           sb.append("average ranks of matched features = ").append(matched.stream()
                   .mapToDouble(FeatureUtility::getRank).average().orElseGet(() -> Double.NaN))
                   .append(", ");

       });






//        if (config.getBoolean("showDetail")){
//            sb.append("matched features and feature ranks = ");
//            for (Pair<String,Integer> pair: pairs){
//                sb.append("(").append(pair.getFirst()).append(",").append(pair.getSecond()).append(")").append(", ");
//            }
//        }

        sb.append("\n");

        return sb.toString();
    }


    public static ClfDataSet loadGoodDataSet(Config config) throws Exception{
        File dataFile = new File(config.getString("input.goodDataSet"),"train.trec");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(dataFile, DataSetType.CLF_SPARSE,true);
        return dataSet;
    }

    public static String showDistribution(int featureRank, String feature, int classIndex, ClfDataSet dataSet,
                                        Map<String, Integer> featureToIndex,Map<Integer, Integer> docToRank,
                                        Config config){
        int featureIndex = featureToIndex.get(feature);
        List<Integer> matchedDocs = IntStream.range(0,dataSet.getNumDataPoints())
                .filter(i -> dataSet.getLabels()[i]==classIndex)
                .filter(i-> dataSet.getColumn(featureIndex).get(i)!=0)
                .mapToObj(i -> i).collect(Collectors.toList());
        //docid, rank
        Comparator<Pair<Integer,Integer>> comparator = Comparator.comparing(Pair::getSecond);
        List<Pair<Integer,Integer>> sorted = matchedDocs.stream().map(i -> new Pair<>(i, docToRank.get(i)))
                .sorted(comparator).collect(Collectors.toList());
        List<Integer> ranks = sorted.stream().map(Pair::getSecond).collect(Collectors.toList());
        SummaryStatistics statistics = new SummaryStatistics();
        for (int rank:ranks){
            statistics.addValue(rank);
        }


        StringBuilder sb = new StringBuilder();
        sb.append(feature).append(", ");
        sb.append("feature rank = ").append(featureRank).append(", ");
        sb.append("number of matched docs = ").append(statistics.getN()).append(", ");
        sb.append("doc rank min = ").append(statistics.getMin()).append(", ");
        sb.append("doc rank max = ").append(statistics.getMax()).append(", ");
        sb.append("doc rank mean = ").append(statistics.getMean()).append(", ");
        sb.append("doc rank std = ").append(statistics.getStandardDeviation()).append("\n");
        if (config.getBoolean("showDetail")){
            for (Pair<Integer,Integer> pair: sorted){
                sb.append("id=").append(dataSet.getDataPointSetting(pair.getFirst()).getExtId()).append(",");
                sb.append("rank=").append(pair.getSecond()).append(", ");
            }
        }


        return sb.toString();

    }

    public static Map<String, Integer> featureToIndex(Config config) throws Exception{
        File dataFile = new File(config.getString("input.goodDataSet"),"train.trec");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(dataFile, DataSetType.CLF_SPARSE,true);
        Map<String, Integer> featureToIndex = new HashMap<>();
        for (int j=0;j<dataSet.getNumFeatures();j++){
            String feature = dataSet.getFeatureSetting(j).getFeatureName();
            featureToIndex.put(feature,j);
        }
        return featureToIndex;
    }

    public static Map<Integer, Integer> docToRank(List<List<Integer>> rankedDocs){
        Map<Integer,Integer> map = new HashMap<>();
        for (List<Integer> list : rankedDocs) {
            for (int i = 0; i < list.size(); i++) {
                int doc = list.get(i);
                map.put(doc, i);
            }
        }
        return map;
    }

    //easy to hard, algorithmId and indexId, each class
    public static List<List<Integer>> rankDocs(Config config) throws Exception{
        File dataFile = new File(config.getString("input.basicDataSet"),"train.trec");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(dataFile, DataSetType.CLF_SPARSE,true);
        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setHistory(5)
                .setGaussianPriorVariance(config.getDouble("gaussianPriorVariance"))
                .setEpsilon(0.1)
                .build();


        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("accuracy on basic dataset = "+ Accuracy.accuracy(logisticRegression,dataSet));
        List<List<Integer>> rankedList = new ArrayList<>();

        List<double[]> probs = logisticRegression.predictClassProbs(dataSet);
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        IntStream.range(0,dataSet.getNumClasses()).forEach(k->
            {List<Integer> list = IntStream.range(0,dataSet.getNumDataPoints())
                    .filter(i-> dataSet.getLabels()[i]==k)
                    .mapToObj(i->new Pair<>(i,probs.get(i)[k])).sorted(comparator.reversed())
                    .map(Pair::getFirst)
                    .collect(Collectors.toList());
                rankedList.add(list);
        });

        return rankedList;
    }

    public static List<List<FeatureUtility>> getGoodNgrams(Config config) throws Exception{
        File dataFile = new File(config.getString("input.goodDataSet"),"train.trec");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(dataFile, DataSetType.CLF_SPARSE,true);
        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setHistory(5)
                .setGaussianPriorVariance(config.getDouble("gaussianPriorVariance"))
                .setEpsilon(0.1)
                .build();


        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("accuracy on good dataset = "+ Accuracy.accuracy(logisticRegression,dataSet));
        int limit = config.getInt("topFeature.limit");
        List<List<FeatureUtility>> goodFeatures = new ArrayList<>();
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            goodFeatures.add(LogisticRegressionInspector.topFeatures(logisticRegression, k, limit));
        }
        return goodFeatures;
    }

    public static void checkNgrams(List<List<FeatureUtility>> features){
        for (int k=0;k<features.size();k++){
            System.out.println("class "+k);
            List<FeatureUtility> list = features.get(k);
            List<FeatureUtility> unigramUtilities = list.stream().filter(util-> util.getName().split(" ").length == 1)
                    .collect(Collectors.toList());
            Set<String> unigrams = unigramUtilities.stream().map(FeatureUtility::getName)
                    .collect(Collectors.toSet());
            System.out.println("number of unigrams = "+unigrams.size());
            System.out.println("sum of unigram weights = "+unigramUtilities.stream().mapToDouble(FeatureUtility::getUtility).sum());

            List<FeatureUtility> ngramUtilities = list.stream().filter(util-> util.getName().split(" ").length > 1)
                    .collect(Collectors.toList());
            System.out.println("sum of ngram weights = "+ngramUtilities.stream().mapToDouble(FeatureUtility::getUtility).sum());
            Set<String> ngrams = ngramUtilities.stream().map(FeatureUtility::getName)
                    .collect(Collectors.toSet());
            System.out.println("number of ngrams = "+ngrams.size());
            List<FeatureUtility> easyNgramUtilities = ngramUtilities.stream().filter(ngram -> isComposedOf(ngram.getName(),unigrams)).collect(Collectors.toList());
            System.out.println("number of easy ngrams = "+easyNgramUtilities.size());
            System.out.println("sum of easy ngram weights = "+easyNgramUtilities.stream().mapToDouble(FeatureUtility::getUtility).sum());

            List<FeatureUtility> hardNgramUtilities = ngramUtilities.stream().filter(ngram -> !isComposedOf(ngram.getName(),unigrams)).collect(Collectors.toList());
            System.out.println("number of hard ngrams = "+hardNgramUtilities.size());
            System.out.println("sum of hard ngram weights = "+hardNgramUtilities.stream().mapToDouble(FeatureUtility::getUtility).sum());

        }
    }

    public static boolean isComposedOf(String ngram, Set<String> unigrams){
        String[] split = ngram.split(" ");
        for (String term: split){
            if (unigrams.contains(term)){
                return true;
            }
        }
        return false;
    }
}
