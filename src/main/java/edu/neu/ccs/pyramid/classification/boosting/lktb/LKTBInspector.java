package edu.neu.ccs.pyramid.classification.boosting.lktb;


import edu.neu.ccs.pyramid.classification.ClassProbability;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.regression.*;
import edu.neu.ccs.pyramid.regression.regression_tree.TreeRule;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;


import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 9/4/14.
 */
public class LKTBInspector {

    //todo: consider newton step and learning rate

    /**
     * only trees are considered
     * @param boosting
     * @param classIndex
     * @return list of feature index and feature name pairs
     */
    public static List<Feature> topFeatures(LKTreeBoost boosting, int classIndex){
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
        List<Feature> list = totalContributions.entrySet().stream().sorted(comparator.reversed())
                .map(Map.Entry::getKey).collect(Collectors.toList());
        return list;
    }


    /**
     *
     * @param lkTreeBoosts ensemble of lktbs
     * @param classIndex
     * @return
     */
    public static List<Feature> topFeatures(List<LKTreeBoost> lkTreeBoosts, int classIndex){
        Map<Feature,Double> totalContributions = new HashMap<>();
        for (LKTreeBoost lkTreeBoost: lkTreeBoosts){
            List<Regressor> regressors = lkTreeBoost.getRegressors(classIndex);
            List<RegressionTree> trees = regressors.stream().filter(regressor ->
                    regressor instanceof RegressionTree)
                    .map(regressor -> (RegressionTree) regressor)
                    .collect(Collectors.toList());
            for (RegressionTree tree: trees){
                Map<Feature,Double> contributions = RegTreeInspector.featureImportance(tree);
                for (Map.Entry<Feature,Double> entry: contributions.entrySet()){
                    Feature feature = entry.getKey();
                    Double contribution = entry.getValue();
                    double oldValue = totalContributions.getOrDefault(feature, 0.0);
                    double newValue = oldValue+contribution;
                    totalContributions.put(feature,newValue);
                }
            }
        }

        Comparator<Map.Entry<Feature,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        List<Feature> list = totalContributions.entrySet().stream().sorted(comparator.reversed())
                .map(Map.Entry::getKey).collect(Collectors.toList());
        return list;
    }

    public static List<Integer> topFeatureIndices(List<LKTreeBoost> lkTreeBoosts, int classIndex){
        return topFeatures(lkTreeBoosts,classIndex).stream().map(Feature::getIndex)
                .collect(Collectors.toList());
    }

    public static List<String> topFeatureNames(List<LKTreeBoost> lkTreeBoosts, int classIndex){
        return topFeatures(lkTreeBoosts,classIndex).stream().map(Feature::getName)
                .collect(Collectors.toList());
    }

    public static Set<Integer> recentlyUsedFeatures(LKTreeBoost boosting, int k){
        Set<Integer> features = new HashSet<>();
        List<Regressor> regressors = boosting.getRegressors(k);
        int size = regressors.size();
        Regressor lastOne = regressors.get(size-1);
        if (lastOne instanceof RegressionTree){
            features.addAll(RegTreeInspector.features((RegressionTree)lastOne));
        }
        return features;
    }

    //todo  speed up
    public static PredictionAnalysis analyzePrediction(LKTreeBoost boosting, ClfDataSet dataSet, int dataPointIndex, int limit){
        PredictionAnalysis predictionAnalysis = new PredictionAnalysis();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        predictionAnalysis.setInternalId(dataPointIndex)
                .setId(idTranslator.toExtId(dataPointIndex))
                .setInternalLabel(dataSet.getLabels()[dataPointIndex])
                .setLabel(labelTranslator.toExtLabel(dataSet.getLabels()[dataPointIndex]));
        int prediction = boosting.predict(dataSet.getRow(dataPointIndex));
        predictionAnalysis.setInternalPrediction(prediction);
        predictionAnalysis.setPrediction(labelTranslator.toExtLabel(prediction));
        double[] probs = boosting.predictClassProbs(dataSet.getRow(dataPointIndex));
        List<ClassProbability> classProbabilities = new ArrayList<>();
        for (int k=0;k<probs.length;k++){
            ClassProbability classProbability = new ClassProbability(k,labelTranslator.toExtLabel(k),probs[k]);
            classProbabilities.add(classProbability);
        }
        predictionAnalysis.setClassProbabilities(classProbabilities);
        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k=0;k<probs.length;k++){
            ClassScoreCalculation classScoreCalculation = decisionProcess(boosting,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,limit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);
        return predictionAnalysis;
    }

    public static ClassScoreCalculation decisionProcess(LKTreeBoost boosting, LabelTranslator labelTranslator, Vector vector, int classIndex, int limit){
        ClassScoreCalculation classScoreCalculation = new ClassScoreCalculation(classIndex,labelTranslator.toExtLabel(classIndex),
                boosting.predictClassScore(vector,classIndex));
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

//    public static String analyzeMistake(LKTreeBoost boosting, Vector vector,
//                                        int trueLabel, int prediction,
//                                        LabelTranslator labelTranslator, int limit){
//        StringBuilder sb = new StringBuilder();
//        sb.append("score for the true label ").append(trueLabel).append("(").append(labelTranslator.toExtLabel(trueLabel))
//                .append(")").append(" = ").append(boosting.predictClassScore(vector,trueLabel)).append("\n");
//
//        sb.append("score for the prediction ").append(prediction).append("(").append(labelTranslator.toExtLabel(prediction))
//                .append(")").append(" = ").append(boosting.predictClassScore(vector,prediction)).append("\n");
//        sb.append("decision process for the true label ").append(trueLabel).append("(").append(labelTranslator.toExtLabel(trueLabel))
//                .append(")").append(":").append("\n");
//        sb.append(decisionProcess(boosting,vector,trueLabel,limit));
//
//        sb.append("decision process for the prediction ").append(prediction).append("(").append(labelTranslator.toExtLabel(prediction))
//                .append(")").append(":").append("\n");
//        sb.append(decisionProcess(boosting,vector,prediction,limit));
//
//        return sb.toString();
//    }
}
