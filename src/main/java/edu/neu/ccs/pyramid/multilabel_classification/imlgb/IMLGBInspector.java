package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.TreeRule;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
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


    public static Map<List<Integer>, Double> countPathMatches(IMLGradientBoosting boosting, DataSet dataSet, int classIndex){
        List<RegressionTree> trees = boosting.getRegressors(classIndex)
                .stream().filter(regressor ->
                regressor instanceof RegressionTree)
                .map(regressor -> (RegressionTree) regressor)
                .collect(Collectors.toList());
        Map<List<Integer>, Double> map = new HashMap<>();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            List<Integer> path = RegTreeInspector.getMatchedPath(trees,dataSet.getRow(i));
            double oldCount = map.getOrDefault(path,0.0);
            map.put(path,oldCount+1);
        }
        return map;
    }

    public static String analyzeMistake(IMLGradientBoosting boosting, Vector vector,
                                        MultiLabel trueLabel, MultiLabel prediction,
                                        LabelTranslator labelTranslator, int limit){
        StringBuilder sb = new StringBuilder();
        List<Integer> difference = MultiLabel.symmetricDifference(trueLabel,prediction).stream().sorted().collect(Collectors.toList());

        double[] classScores = boosting.calClassScores(vector);
        sb.append("score for the true labels ").append(trueLabel)
                .append("(").append(trueLabel.toStringWithExtLabels(labelTranslator)).append(") = ");
        sb.append(boosting.calAssignmentScore(trueLabel,classScores)).append("\n");

        sb.append("score for the predicted labels ").append(prediction)
                .append("(").append(prediction.toStringWithExtLabels(labelTranslator)).append(") = ");;
        sb.append(boosting.calAssignmentScore(prediction,classScores)).append("\n");

        for (int k: difference){
            sb.append("score for class ").append(k).append("(").append(labelTranslator.toExtLabel(k)).append(")")
                    .append(" =").append(classScores[k]).append("\n");
        }

        for (int k: difference){
            sb.append("decision process for class ").append(k).append("(").append(labelTranslator.toExtLabel(k)).append("):\n");
            sb.append(decisionProcess(boosting,vector,k,limit));
            sb.append("--------------------------------------------------").append("\n");
        }

        return sb.toString();
    }

    public static String decisionProcess(IMLGradientBoosting boosting, Vector vector, int classIndex, int limit){
        StringBuilder sb = new StringBuilder();
        List<Regressor> regressors = boosting.getRegressors(classIndex).stream().collect(Collectors.toList());
        List<TreeRule> treeRules = new ArrayList<>();
        for (int i=0;i<regressors.size();i++){
            Regressor regressor = regressors.get(i);
            if (regressor instanceof ConstantRegressor){
                sb.append("prior score for the class = ");
                sb.append(((ConstantRegressor) regressor).getScore()).append("\n");
            }

            if (regressor instanceof RegressionTree){
                RegressionTree tree = (RegressionTree)regressor;
                TreeRule treeRule = new TreeRule(tree,vector);
                treeRules.add(treeRule);
            }
        }
        Comparator<TreeRule> comparator = Comparator.comparing(decision -> Math.abs(decision.getScore()));
        List<TreeRule> merged = TreeRule.merge(treeRules).stream().sorted(comparator.reversed())
                .limit(limit).collect(Collectors.toList());
        for (TreeRule treeRule : merged){
            sb.append(treeRule.toString());
            sb.append("\n");
        }

        return sb.toString();
    }
}
