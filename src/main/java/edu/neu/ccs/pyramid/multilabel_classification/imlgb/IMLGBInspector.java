package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.feature_selection.FeatureDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGradientBoosting;
import edu.neu.ccs.pyramid.regression.*;
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
        List<Feature> list = totalContributions.entrySet().stream().sorted(comparator.reversed()).limit(limit)
                .map(Map.Entry::getKey).collect(Collectors.toList());
        TopFeatures topFeatures = new TopFeatures();
        topFeatures.setTopFeatures(list);
        topFeatures.setClassIndex(classIndex);
        LabelTranslator labelTranslator = boosting.getLabelTranslator();
        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));
        return topFeatures;
    }


    public static TopFeatures topFeatures(IMLGradientBoosting boosting,  int classIndex, int limit,Collection<FeatureDistribution> inputDistributions){
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
        List<Feature> list = totalContributions.entrySet().stream().sorted(comparator.reversed()).limit(limit)
                .map(Map.Entry::getKey).collect(Collectors.toList());
        TopFeatures topFeatures = new TopFeatures();
        topFeatures.setTopFeatures(list);
        topFeatures.setClassIndex(classIndex);
        LabelTranslator labelTranslator = boosting.getLabelTranslator();
        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));

        List<FeatureDistribution> featureDistributions = new ArrayList<>();


//        for (Feature feature: list){
//            feature.clearIndex();
//        }

        for (Feature feature: list){
            if (feature instanceof Ngram){
                FeatureDistribution featureDistribution = null;
                for (FeatureDistribution distribution: inputDistributions){
                    distribution.getFeature().setIndex(feature.getIndex());
                    if (distribution.getFeature().equals(feature)){
                        featureDistribution = distribution;
                        break;
                    }
                }
                featureDistributions.add(featureDistribution);
            }
        }

        topFeatures.setFeatureDistributions(featureDistributions);
        return topFeatures;
    }


//    /**
//     *
//     * @param boostings ensemble of lktbs
//     * @param classIndex
//     * @return
//     */
//    public static List<Pair<Integer,String>> topFeatures(List<IMLGradientBoosting> boostings, int classIndex){
//        Map<Integer,Pair<String,Double>> totalContributions = new HashMap<>();
//        for (IMLGradientBoosting boosting: boostings){
//            List<Regressor> regressors = boosting.getRegressors(classIndex);
//            List<RegressionTree> trees = regressors.stream().filter(regressor ->
//                    regressor instanceof RegressionTree)
//                    .map(regressor -> (RegressionTree) regressor)
//                    .collect(Collectors.toList());
//            for (RegressionTree tree: trees){
//                Map<Integer,Pair<String,Double>> contributions = RegTreeInspector.featureImportance(tree);
//                for (Map.Entry<Integer, Pair<String,Double>> entry: contributions.entrySet()){
//                    int featureIndex = entry.getKey();
//                    Pair<String,Double> pair = entry.getValue();
//                    String featureName = pair.getFirst();
//                    Pair<String,Double> oldPair = totalContributions.getOrDefault(featureIndex,new Pair<>(featureName,0.0));
//                    Pair<String,Double> newPair = new Pair<>(featureName,oldPair.getSecond()+pair.getSecond());
//                    totalContributions.put(featureIndex,newPair);
//                }
//            }
//        }
//
//        Comparator<Map.Entry<Integer, Pair<String,Double>>> comparator = Comparator.comparing(entry -> entry.getValue().getSecond());
//        List<Pair<Integer,String>> list = totalContributions.entrySet().stream().sorted(comparator.reversed())
//                .map(entry -> new Pair<>(entry.getKey(),entry.getValue().getFirst())).collect(Collectors.toList());
//        return list;
//    }
//
//    public static List<Integer> topFeatureIndices(List<IMLGradientBoosting> boostings, int classIndex){
//        return topFeatures(boostings,classIndex).stream().map(Pair::getFirst)
//                .collect(Collectors.toList());
//    }
//
//    public static List<String> topFeatureNames(List<IMLGradientBoosting> boostings, int classIndex){
//        return topFeatures(boostings,classIndex).stream().map(Pair::getSecond)
//                .collect(Collectors.toList());
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






    public static ClassScoreCalculation decisionProcess(IMLGradientBoosting boosting, LabelTranslator labelTranslator,
                                                        Vector vector, int classIndex, int limit){
        ClassScoreCalculation classScoreCalculation = new ClassScoreCalculation(classIndex,labelTranslator.toExtLabel(classIndex),
                boosting.predictClassScore(vector,classIndex));
        double prob = boosting.predictClassProb(vector,classIndex);
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


    //todo  speed up


    public static MultiLabelPredictionAnalysis analyzePrediction(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet,
                                                                 int dataPointIndex, List<Integer> classes, int limit){
        MultiLabelPredictionAnalysis predictionAnalysis = new MultiLabelPredictionAnalysis();
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        predictionAnalysis.setInternalId(dataPointIndex);
        predictionAnalysis.setId(idTranslator.toExtId(dataPointIndex));
        predictionAnalysis.setInternalLabels(dataSet.getMultiLabels()[dataPointIndex].getMatchedLabelsOrdered());
        List<String> labels = dataSet.getMultiLabels()[dataPointIndex].getMatchedLabelsOrdered().stream()
                .map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setLabels(labels);
        predictionAnalysis.setProbForTrueLabels(boosting.predictAssignmentProb(dataSet.getRow(dataPointIndex),
                dataSet.getMultiLabels()[dataPointIndex]));

        MultiLabel predictedLabels = boosting.predict(dataSet.getRow(dataPointIndex));
        List<Integer> internalPrediction = predictedLabels.getMatchedLabelsOrdered();
        predictionAnalysis.setInternalPrediction(internalPrediction);
        List<String> prediction = internalPrediction.stream().map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setPrediction(prediction);
        predictionAnalysis.setProbForPredictedLabels(boosting.predictAssignmentProb(dataSet.getRow(dataPointIndex),predictedLabels));

        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k: classes){
            ClassScoreCalculation classScoreCalculation = decisionProcess(boosting,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,limit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);

        List<MultiLabelPredictionAnalysis.ClassRankInfo> ranking = classes.stream().map(label -> {
                    MultiLabelPredictionAnalysis.ClassRankInfo rankInfo = new MultiLabelPredictionAnalysis.ClassRankInfo();
                    rankInfo.setClassIndex(label);
                    rankInfo.setClassName(labelTranslator.toExtLabel(label));
                    rankInfo.setProb(boosting.predictClassProb(dataSet.getRow(dataPointIndex), label));
                    return rankInfo;
                }
            ).collect(Collectors.toList());
            predictionAnalysis.setPredictedRanking(ranking);

            return predictionAnalysis;
        }
    }
