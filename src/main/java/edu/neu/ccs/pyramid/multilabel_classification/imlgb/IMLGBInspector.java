package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by chengli on 10/13/14.
 */
public class IMLGBInspector {
    //todo: consider newton step and learning rate

    /**
     * only trees are considered
     * @param boosting
     * @param classIndex
     * @return list of feature index and feature name pairs
     */
    public static List<Pair<Integer,String>> topFeatures(IMLGradientBoosting boosting, int classIndex){
        Map<Integer,Pair<String,Double>> totalContributions = new HashMap<>();
        List<Regressor> regressors = boosting.getRegressors(classIndex);
        List<RegressionTree> trees = regressors.stream().filter(regressor ->
                regressor instanceof RegressionTree)
                .map(regressor -> (RegressionTree) regressor)
                .collect(Collectors.toList());
        for (RegressionTree tree: trees){
            Map<Integer,Pair<String,Double>> contributions = RegTreeInspector.featureImportance(tree);
            for (Map.Entry<Integer, Pair<String,Double>> entry: contributions.entrySet()){
                int featureIndex = entry.getKey();
                Pair<String,Double> pair = entry.getValue();
                String featureName = pair.getFirst();
                Pair<String,Double> oldPair = totalContributions.getOrDefault(featureIndex,new Pair<>(featureName,0.0));
                Pair<String,Double> newPair = new Pair<>(featureName,oldPair.getSecond()+pair.getSecond());
                totalContributions.put(featureIndex,newPair);
            }
        }
        Comparator<Map.Entry<Integer, Pair<String,Double>>> comparator = Comparator.comparing(entry -> entry.getValue().getSecond());
        List<Pair<Integer,String>> list = totalContributions.entrySet().stream().sorted(comparator.reversed())
                .map(entry -> new Pair<>(entry.getKey(),entry.getValue().getFirst())).collect(Collectors.toList());
        return list;
    }

    public static List<Integer> topFeatureIndices(IMLGradientBoosting boosting, int classIndex){
        return topFeatures(boosting,classIndex).stream().map(Pair::getFirst)
                .collect(Collectors.toList());
    }

    public static List<String> topFeatureNames(IMLGradientBoosting boosting, int classIndex){
        return topFeatures(boosting,classIndex).stream().map(Pair::getSecond)
                .collect(Collectors.toList());
    }

    /**
     *
     * @param boostings ensemble of lktbs
     * @param classIndex
     * @return
     */
    public static List<Pair<Integer,String>> topFeatures(List<IMLGradientBoosting> boostings, int classIndex){
        Map<Integer,Pair<String,Double>> totalContributions = new HashMap<>();
        for (IMLGradientBoosting boosting: boostings){
            List<Regressor> regressors = boosting.getRegressors(classIndex);
            List<RegressionTree> trees = regressors.stream().filter(regressor ->
                    regressor instanceof RegressionTree)
                    .map(regressor -> (RegressionTree) regressor)
                    .collect(Collectors.toList());
            for (RegressionTree tree: trees){
                Map<Integer,Pair<String,Double>> contributions = RegTreeInspector.featureImportance(tree);
                for (Map.Entry<Integer, Pair<String,Double>> entry: contributions.entrySet()){
                    int featureIndex = entry.getKey();
                    Pair<String,Double> pair = entry.getValue();
                    String featureName = pair.getFirst();
                    Pair<String,Double> oldPair = totalContributions.getOrDefault(featureIndex,new Pair<>(featureName,0.0));
                    Pair<String,Double> newPair = new Pair<>(featureName,oldPair.getSecond()+pair.getSecond());
                    totalContributions.put(featureIndex,newPair);
                }
            }
        }

        Comparator<Map.Entry<Integer, Pair<String,Double>>> comparator = Comparator.comparing(entry -> entry.getValue().getSecond());
        List<Pair<Integer,String>> list = totalContributions.entrySet().stream().sorted(comparator.reversed())
                .map(entry -> new Pair<>(entry.getKey(),entry.getValue().getFirst())).collect(Collectors.toList());
        return list;
    }

    public static List<Integer> topFeatureIndices(List<IMLGradientBoosting> boostings, int classIndex){
        return topFeatures(boostings,classIndex).stream().map(Pair::getFirst)
                .collect(Collectors.toList());
    }

    public static List<String> topFeatureNames(List<IMLGradientBoosting> boostings, int classIndex){
        return topFeatures(boostings,classIndex).stream().map(Pair::getSecond)
                .collect(Collectors.toList());
    }
}
