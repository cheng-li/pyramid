package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.sentiment_analysis.Negation;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * deal with negations in a dataset
 * Created by chengli on 2/12/15.
 */
public class Exp67 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input,"test.trec"),
                DataSetType.CLF_SPARSE, true);
        Map<String, Integer> featureToIndex = featureToIndex(dataSet);
        List<Pair<Integer, Integer>> negationPairs = getNegationPairs(featureToIndex);
        showPairs(dataSet,negationPairs);

        ClfDataSet newTrain = subtract(dataSet,negationPairs);
        TRECFormat.save(newTrain,new File(config.getString("output.folder"),"train.trec"));

        ClfDataSet newTest = subtract(testSet,negationPairs);
        TRECFormat.save(newTest,new File(config.getString("output.folder"),"test.trec"));

    }

    public static Map<String, Integer> featureToIndex(DataSet dataSet){
        Map<String, Integer> map = new HashMap<>();
        for (int i=0;i<dataSet.getNumFeatures();i++){
            String feature = dataSet.getFeatureSetting(i).getFeatureName()
                    .replaceAll("\\(slop=.\\)", "");
            map.put(feature,i);
        }
        return map;
    }

    //todo more robust
    public static List<Pair<Integer, Integer>> getNegationPairs(Map<String, Integer> featureToIndex){
        List<Pair<Integer, Integer>> list = new ArrayList<>();
        featureToIndex.entrySet().stream().filter(entry->
                Negation.startsWithNegation(entry.getKey())||Negation.endsWithNegation(entry.getKey()))
                .forEach(entry -> {
                    String negated = entry.getKey();
                    int negationIndex = entry.getValue();
                    String original = Negation.removeNegation(negated);
                    if (featureToIndex.containsKey(original)){
                        int originalIndex = featureToIndex.get(original);
                        Pair<Integer,Integer> pair = new Pair<>(negationIndex,originalIndex);
                        list.add(pair);
                    }
                });
        return list;
    }

    public static void showPairs(DataSet dataSet, List<Pair<Integer, Integer>> pairs){
        List<Pair<String,String>> list = pairs.stream().map(pair -> new Pair<>(dataSet.getFeatureSetting(pair.getFirst()).getFeatureName()
                ,dataSet.getFeatureSetting(pair.getSecond()).getFeatureName())).collect(Collectors.toList());
        System.out.println(list);
    }


    //original - negation
    public static ClfDataSet subtract(ClfDataSet dataSet, List<Pair<Integer, Integer>> pairs){
        for (Pair<Integer, Integer> pair: pairs){
            int negation = pair.getFirst();
            int original = pair.getSecond();
//            Vector diff = dataSet.getColumn(original).minus(dataSet.getColumn(negation));
            for (Vector.Element element:dataSet.getColumn(negation).nonZeroes()){
                int index = element.index();
                dataSet.setFeatureValue(index,original,0);
            }
        }
        return dataSet;
    }
}
