package edu.neu.ccs.pyramid.core.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.core.dataset.*;
import edu.neu.ccs.pyramid.core.feature.TopFeatures;
import edu.neu.ccs.pyramid.core.regression.*;
import edu.neu.ccs.pyramid.core.regression.regression_tree.TreeRule;
import edu.neu.ccs.pyramid.core.feature.Feature;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegressionTree;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public class HMLGBInspector {
    //todo: consider newton step and learning rate

    /**
     * only trees are considered
     * @param boosting
     * @param classIndex
     * @return list of feature index and feature name pairs
     */
    public static TopFeatures topFeatures(HMLGradientBoosting boosting, int classIndex, int limit){
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


//    /**
//     *
//     * @param boostings ensemble of lktbs
//     * @param classIndex
//     * @return
//     */
//    public static List<Pair<Integer,String>> topFeatures(List<HMLGradientBoosting> boostings, int classIndex){
//        Map<Integer,Pair<String,Double>> totalContributions = new HashMap<>();
//        for (HMLGradientBoosting boosting: boostings){
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
//    public static List<Integer> topFeatureIndices(List<HMLGradientBoosting> boostings, int classIndex){
//        return topFeatures(boostings,classIndex).stream().map(Pair::getFirst)
//                .collect(Collectors.toList());
//    }
//
//    public static List<String> topFeatureNames(List<HMLGradientBoosting> boostings, int classIndex){
//        return topFeatures(boostings,classIndex).stream().map(Pair::getSecond)
//                .collect(Collectors.toList());
//    }


    public static Map<List<Integer>, Double> countPathMatches(HMLGradientBoosting boosting, DataSet dataSet, int classIndex){
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
//
//    public static String analyzeMistake(HMLGradientBoosting boosting, Vector vector,
//                                        MultiLabel trueLabel, MultiLabel prediction,
//                                        LabelTranslator labelTranslator, int limit){
//        StringBuilder sb = new StringBuilder();
//        List<Integer> difference = MultiLabel.symmetricDifference(trueLabel,prediction).stream().sorted().collect(Collectors.toList());
//
//        double[] classScores = boosting.predictClassScores(vector);
//        sb.append("score for the true labels ").append(trueLabel)
//                .append("(").append(trueLabel.toStringWithExtLabels(labelTranslator)).append(") = ");
//        sb.append(boosting.calAssignmentScore(trueLabel,classScores)).append("\n");
//
//        sb.append("score for the predicted labels ").append(prediction)
//                .append("(").append(prediction.toStringWithExtLabels(labelTranslator)).append(") = ");;
//        sb.append(boosting.calAssignmentScore(prediction,classScores)).append("\n");
//
//        for (int k: difference){
//            sb.append("score for class ").append(k).append("(").append(labelTranslator.toExtLabel(k)).append(")")
//                    .append(" =").append(classScores[k]).append("\n");
//        }
//
//        for (int k: difference){
//            sb.append("decision process for class ").append(k).append("(").append(labelTranslator.toExtLabel(k)).append("):\n");
//            sb.append(decisionProcess(boosting,vector,k,limit));
//            sb.append("--------------------------------------------------").append("\n");
//        }
//
//        return sb.toString();
//    }

    public static ClassScoreCalculation decisionProcess(HMLGradientBoosting boosting, LabelTranslator labelTranslator,
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

    public static MultiLabelPredictionAnalysis analyzePrediction(HMLGradientBoosting boosting, MultiLabelClfDataSet dataSet,
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
        predictionAnalysis.setProbForPredictedLabels(boosting.predictAssignmentProb(dataSet.getRow(dataPointIndex), predictedLabels));

        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k: classes){
            ClassScoreCalculation classScoreCalculation = decisionProcess(boosting,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,limit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);
        return predictionAnalysis;
    }

    public static MultiLabelPredictionAnalysis analyzePrediction(HMLGradientBoosting boosting, MultiLabelClfDataSet dataSet,
                                                                 int dataPointIndex, int limit){
        List<Integer> classes = IntStream.range(0,boosting.getNumClasses()).mapToObj(i-> i).collect(Collectors.toList());
        return analyzePrediction(boosting,dataSet,dataPointIndex,classes,limit);
    }
}
