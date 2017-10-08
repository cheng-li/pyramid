package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
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

    public static List<LabelModel> getAllRules(IMLGradientBoosting boosting){
        List<LabelModel> labelModels = new ArrayList<>();
        for (int k=0;k<boosting.getNumClasses();k++){
            labelModels.add(new LabelModel(boosting, k));
        }
        return labelModels;
    }


    //todo  speed up


    public static  MultiLabelPredictionAnalysis analyzePrediction(IMLGradientBoosting boosting,
                                                                  PluginPredictor<IMLGradientBoosting> pluginPredictor,
                                                                 MultiLabelClfDataSet dataSet,
                                                                 int dataPointIndex,  int ruleLimit,
                                                                 int labelSetLimit, double classProbThreshold){
        MultiLabelPredictionAnalysis predictionAnalysis = new MultiLabelPredictionAnalysis();
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        predictionAnalysis.setInternalId(dataPointIndex);
        predictionAnalysis.setId(idTranslator.toExtId(dataPointIndex));
        predictionAnalysis.setInternalLabels(dataSet.getMultiLabels()[dataPointIndex].getMatchedLabelsOrdered());
        List<String> labels = dataSet.getMultiLabels()[dataPointIndex].getMatchedLabelsOrdered().stream()
                .map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setLabels(labels);
        if (pluginPredictor instanceof SubsetAccPredictor || pluginPredictor instanceof InstanceF1Predictor){
            predictionAnalysis.setProbForTrueLabels(boosting.predictAssignmentProbWithConstraint(dataSet.getRow(dataPointIndex),
                    dataSet.getMultiLabels()[dataPointIndex]));
        }

        if (pluginPredictor instanceof HammingPredictor || pluginPredictor instanceof MacroF1Predictor){
            predictionAnalysis.setProbForTrueLabels(boosting.predictAssignmentProbWithoutConstraint(dataSet.getRow(dataPointIndex),
                    dataSet.getMultiLabels()[dataPointIndex]));
        }


        MultiLabel predictedLabels = pluginPredictor.predict(dataSet.getRow(dataPointIndex));
        List<Integer> internalPrediction = predictedLabels.getMatchedLabelsOrdered();
        predictionAnalysis.setInternalPrediction(internalPrediction);
        List<String> prediction = internalPrediction.stream().map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setPrediction(prediction);

        if (pluginPredictor instanceof SubsetAccPredictor || pluginPredictor instanceof InstanceF1Predictor){
            predictionAnalysis.setProbForPredictedLabels(boosting.predictAssignmentProbWithConstraint(dataSet.getRow(dataPointIndex),predictedLabels));
        }

        if (pluginPredictor instanceof HammingPredictor || pluginPredictor instanceof MacroF1Predictor ){
            predictionAnalysis.setProbForPredictedLabels(boosting.predictAssignmentProbWithoutConstraint(dataSet.getRow(dataPointIndex),predictedLabels));
        }

        double[] classProbs = boosting.predictClassProbs(dataSet.getRow(dataPointIndex));
        List<Integer> classes = new ArrayList<Integer>();
        for (int k = 0; k < boosting.getNumClasses(); k++){
            if (classProbs[k]>=classProbThreshold
                    ||dataSet.getMultiLabels()[dataPointIndex].matchClass(k)
                    ||predictedLabels.matchClass(k)){
                classes.add(k);
            }
        }

        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k: classes){
            ClassScoreCalculation classScoreCalculation = decisionProcess(boosting,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,ruleLimit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);

        List<MultiLabelPredictionAnalysis.ClassRankInfo> labelRanking = classes.stream().map(label -> {
                    MultiLabelPredictionAnalysis.ClassRankInfo rankInfo = new MultiLabelPredictionAnalysis.ClassRankInfo();
                    rankInfo.setClassIndex(label);
                    rankInfo.setClassName(labelTranslator.toExtLabel(label));
                    rankInfo.setProb(boosting.predictClassProb(dataSet.getRow(dataPointIndex), label));
                    return rankInfo;
                }
            ).collect(Collectors.toList());
            predictionAnalysis.setPredictedRanking(labelRanking);


        List<MultiLabelPredictionAnalysis.LabelSetProbInfo> labelSetRanking = null;

        if (pluginPredictor instanceof SubsetAccPredictor || pluginPredictor instanceof InstanceF1Predictor){
            double[] labelSetProbs = boosting.predictAllAssignmentProbsWithConstraint(dataSet.getRow(dataPointIndex));
            labelSetRanking = IntStream.range(0,boosting.getAssignments().size())
            .mapToObj(i -> {
                MultiLabel multiLabel = boosting.getAssignments().get(i);
                double setProb = labelSetProbs[i];
                MultiLabelPredictionAnalysis.LabelSetProbInfo labelSetProbInfo = new MultiLabelPredictionAnalysis.LabelSetProbInfo(multiLabel, setProb, labelTranslator);
                return labelSetProbInfo;
            }).sorted(Comparator.comparing(MultiLabelPredictionAnalysis.LabelSetProbInfo::getProbability).reversed())
                    .limit(labelSetLimit)
                    .collect(Collectors.toList());
        }


        if (pluginPredictor instanceof HammingPredictor || pluginPredictor instanceof MacroF1Predictor){
            labelSetRanking = new ArrayList<>();
            DynamicProgramming dp = new DynamicProgramming(classProbs);
            for (int c=0;c<labelSetLimit;c++){
                DynamicProgramming.Candidate candidate = dp.nextHighest();
                MultiLabel multiLabel = candidate.getMultiLabel();
                double setProb = candidate.getProbability();
                MultiLabelPredictionAnalysis.LabelSetProbInfo labelSetProbInfo = new MultiLabelPredictionAnalysis.LabelSetProbInfo(multiLabel, setProb, labelTranslator);
                labelSetRanking.add(labelSetProbInfo);
            }
        }

        predictionAnalysis.setPredictedLabelSetRanking(labelSetRanking);

        return predictionAnalysis;
    }


    public static  String simplePredictionAnalysis(IMLGradientBoosting boosting,
                                                   PluginPredictor<IMLGradientBoosting> pluginPredictor,
                                                                             MultiLabelClfDataSet dataSet,
                                                  int dataPointIndex,  double classProbThreshold){
        StringBuilder sb = new StringBuilder();
        MultiLabel trueLabels = dataSet.getMultiLabels()[dataPointIndex];
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        double[] classProbs = boosting.predictClassProbs(dataSet.getRow(dataPointIndex));
        MultiLabel predicted = pluginPredictor.predict(dataSet.getRow(dataPointIndex));

        List<Integer> classes = new ArrayList<Integer>();
        for (int k = 0; k < boosting.getNumClasses(); k++){
            if (classProbs[k]>=classProbThreshold
                    ||dataSet.getMultiLabels()[dataPointIndex].matchClass(k)
                    ||predicted.matchClass(k)){
                classes.add(k);
            }
        }

        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        List<Pair<Integer,Double>> list = classes.stream().map(l -> new Pair<Integer, Double>(l, classProbs[l]))
                .sorted(comparator.reversed()).collect(Collectors.toList());
        for (Pair<Integer,Double> pair: list){
            int label = pair.getFirst();
            double prob = pair.getSecond();
            int match = 0;
            if (trueLabels.matchClass(label)){
                match=1;
            }
            sb.append(id).append("\t").append(labelTranslator.toExtLabel(label)).append("\t")
                    .append("single").append("\t").append(prob)
                    .append("\t").append(match).append("\n");
        }
       

        double probability = 0;
        if (pluginPredictor instanceof SubsetAccPredictor || pluginPredictor instanceof InstanceF1Predictor){
            probability = boosting.predictAssignmentProbWithConstraint(dataSet.getRow(dataPointIndex),predicted);
        }

        if (pluginPredictor instanceof HammingPredictor || pluginPredictor instanceof MacroF1Predictor){
            probability = boosting.predictAssignmentProbWithoutConstraint(dataSet.getRow(dataPointIndex),predicted);
        }

        List<Integer> predictedList = predicted.getMatchedLabelsOrdered();
        sb.append(id).append("\t");
        for (int i=0;i<predictedList.size();i++){
            sb.append(labelTranslator.toExtLabel(predictedList.get(i)));
            if (i!=predictedList.size()-1){
                sb.append(",");
            }
        }
        sb.append("\t");
        int setMatch = 0;
        if (predicted.equals(trueLabels)){
            setMatch=1;
        }
        sb.append("set").append("\t").append(probability).append("\t").append(setMatch).append("\n");
        return sb.toString();
    }

}
