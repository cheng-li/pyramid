package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.br.SupportPredictor;
import edu.neu.ccs.pyramid.regression.*;
import edu.neu.ccs.pyramid.regression.regression_tree.TreeRule;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/13/14.
 */
public class IMLGBInspector {

    /**
     *
     * @param boosting
     * @param classIndex
     * @return features selected by the model
     */
    public static Set<Integer> getSelectedFeatures(IMLGradientBoosting boosting, int classIndex){
        List<Regressor> regressors = boosting.getRegressors(classIndex);
        List<RegressionTree> trees = regressors.stream().filter(regressor ->
                regressor instanceof RegressionTree)
                .map(regressor -> (RegressionTree) regressor)
                .collect(Collectors.toList());
        Set<Integer> features = new HashSet<>();
        for (RegressionTree tree: trees){
            features.addAll(RegTreeInspector.features(tree));
        }
        return features;
    }


    public static Set<Integer> getSelectedFeatures(IMLGradientBoosting boosting){

        Set<Integer> features = new HashSet<>();
        for (int i=0;i<boosting.getNumClasses();i++){
            features.addAll(getSelectedFeatures(boosting, i));
        }
        return features;
    }


    //todo: consider newton step and learning rate

    /**
     * only trees are considered
     * @param boosting
     * @param classIndex
     * @return list of feature index and feature name pairs
     */
    public static TopFeatures topFeatures(IMLGradientBoosting boosting, int classIndex, int limit){
        Map<Feature,Double> totalContributions = new HashMap<>();
        List<Regressor> regressors = boosting.getRegressors(classIndex);
        List<RegressionTree> trees = regressors.stream().filter(regressor ->
                regressor instanceof RegressionTree)
                .map(regressor -> (RegressionTree) regressor)
                .collect(Collectors.toList());
        for (RegressionTree tree: trees){
            Map<Feature,Double> contributions = RegTreeInspector.featureImportance(tree);
            for (Map.Entry<Feature,Double> entry: contributions.entrySet()){
                Feature feature = entry.getKey();
                Double contribution = entry.getValue();
                double oldValue = totalContributions.getOrDefault(feature,0.0);
                double newValue = oldValue+contribution;
                totalContributions.put(feature,newValue);
            }
        }
        Comparator<Map.Entry<Feature,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        List<Map.Entry<Feature,Double>> topKList = totalContributions.entrySet().stream().sorted(comparator.reversed()).limit(limit).collect(Collectors.toList());
        List<Feature> list = topKList.stream().map(Map.Entry::getKey).collect(Collectors.toList());
        List<Double> utilities = topKList.stream().map(Map.Entry::getValue).collect(Collectors.toList());
        TopFeatures topFeatures = new TopFeatures();
        topFeatures.setUtilities(utilities);
        topFeatures.setTopFeatures(list);
        topFeatures.setClassIndex(classIndex);
        LabelTranslator labelTranslator = boosting.getLabelTranslator();
        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));
        return topFeatures;
    }


//    public static TopFeatures topFeatures(IMLGradientBoosting boosting,  int classIndex, int limit,Collection<FeatureDistribution> inputDistributions){
//        Map<Feature,Double> totalContributions = new HashMap<>();
//        List<Regressor> regressors = boosting.getRegressors(classIndex);
//        List<RegressionTree> trees = regressors.stream().filter(regressor ->
//                regressor instanceof RegressionTree)
//                .map(regressor -> (RegressionTree) regressor)
//                .collect(Collectors.toList());
//        for (RegressionTree tree: trees){
//            Map<Feature,Double> contributions = RegTreeInspector.featureImportance(tree);
//            for (Map.Entry<Feature,Double> entry: contributions.entrySet()){
//                Feature feature = entry.getKey();
//                Double contribution = entry.getValue();
//                double oldValue = totalContributions.getOrDefault(feature,0.0);
//                double newValue = oldValue+contribution;
//                totalContributions.put(feature,newValue);
//            }
//        }
//        Comparator<Map.Entry<Feature,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
//        List<Feature> list = totalContributions.entrySet().stream().sorted(comparator.reversed()).limit(limit)
//                .map(Map.Entry::getKey).collect(Collectors.toList());
//        TopFeatures topFeatures = new TopFeatures();
//        topFeatures.setTopFeatures(list);
//        topFeatures.setClassIndex(classIndex);
//        LabelTranslator labelTranslator = boosting.getLabelTranslator();
//        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));
//
//        List<FeatureDistribution> featureDistributions = new ArrayList<>();
//
//
////        for (Feature feature: list){
////            feature.clearIndex();
////        }
//
//        for (Feature feature: list){
//            if (feature instanceof Ngram){
//                FeatureDistribution featureDistribution = null;
//                for (FeatureDistribution distribution: inputDistributions){
//                    distribution.getFeature().setIndex(feature.getIndex());
//                    if (distribution.getFeature().equals(feature)){
//                        featureDistribution = distribution;
//                        break;
//                    }
//                }
//                featureDistributions.add(featureDistribution);
//            }
//        }
//
//        topFeatures.setFeatureDistributions(featureDistributions);
//        return topFeatures;
//    }


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


    public static ClassScoreCalculation decisionProcess(IMLGradientBoosting boosting, LabelTranslator labelTranslator, double prob,
                                                        Vector vector, int classIndex, int limit){
        ClassScoreCalculation classScoreCalculation = new ClassScoreCalculation(classIndex,labelTranslator.toExtLabel(classIndex),
                boosting.predictClassScore(vector,classIndex));
        classScoreCalculation.setClassProbability(prob);
        List<Regressor> regressors = boosting.getRegressors(classIndex);
        List<TreeRule> treeRules = new ArrayList<>();
        for (Regressor regressor : regressors) {
            if (regressor instanceof ConstantRegressor) {
                Rule rule = new ConstantRule(((ConstantRegressor) regressor).getScore());
                classScoreCalculation.addRule(rule);
            }

            if (regressor instanceof RegressionTree) {
                RegressionTree tree = (RegressionTree) regressor;
                TreeRule treeRule = new TreeRule(tree, vector);
                treeRules.add(treeRule);
            }
        }
        Comparator<TreeRule> comparator = Comparator.comparing(decision -> Math.abs(decision.getScore()));
        List<TreeRule> merged = TreeRule.merge(treeRules).stream().sorted(comparator.reversed())
                .limit(limit).collect(Collectors.toList());
        for (TreeRule treeRule : merged){
            classScoreCalculation.addRule(treeRule);
        }

        return classScoreCalculation;
    }

    public static List<LabelModel> getAllRules(IMLGradientBoosting boosting){
        return IntStream.range(0, boosting.getNumClasses()).parallel()
                .mapToObj(k->new LabelModel(boosting, k)).collect(Collectors.toList());

    }







}
