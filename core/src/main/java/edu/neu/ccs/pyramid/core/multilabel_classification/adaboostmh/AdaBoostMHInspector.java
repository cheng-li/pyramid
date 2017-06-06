package edu.neu.ccs.pyramid.core.multilabel_classification.adaboostmh;

import edu.neu.ccs.pyramid.core.dataset.IdTranslator;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.feature.Feature;
import edu.neu.ccs.pyramid.core.feature.TopFeatures;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.core.regression.*;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeInspector;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.core.regression.regression_tree.TreeRule;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 4/4/15.
 */
public class AdaBoostMHInspector {


    public static TopFeatures topFeatures(AdaBoostMH boosting, int classIndex, int limit){
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

    public static ClassScoreCalculation decisionProcess(AdaBoostMH boosting, MultiLabelClassifier.ClassProbEstimator scaling,
                                                        LabelTranslator labelTranslator,
                                                        Vector vector, int classIndex, int limit){
        ClassScoreCalculation classScoreCalculation = new ClassScoreCalculation(classIndex,labelTranslator.toExtLabel(classIndex),
                boosting.predictClassScore(vector,classIndex));
        double prob = scaling.predictClassProb(vector,classIndex);
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

    /**
     * can be binary scaling or across-class scaling
     * @param boosting
     * @param scaling
     * @param dataSet
     * @param dataPointIndex
     * @param classes
     * @param limit
     * @return
     */
    public static MultiLabelPredictionAnalysis analyzePrediction(AdaBoostMH boosting, MultiLabelClassifier.ClassProbEstimator  scaling,
                                                                 MultiLabelClfDataSet dataSet,
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
        double probForTrueLabels = Double.NaN;
        if (scaling instanceof MultiLabelClassifier.AssignmentProbEstimator){

            probForTrueLabels = ((MultiLabelClassifier.AssignmentProbEstimator) scaling).predictAssignmentProb(dataSet.getRow(dataPointIndex),
                    dataSet.getMultiLabels()[dataPointIndex]);
        }
        predictionAnalysis.setProbForTrueLabels(probForTrueLabels);

        MultiLabel predictedLabels = boosting.predict(dataSet.getRow(dataPointIndex));
        List<Integer> internalPrediction = predictedLabels.getMatchedLabelsOrdered();
        predictionAnalysis.setInternalPrediction(internalPrediction);
        List<String> prediction = internalPrediction.stream().map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setPrediction(prediction);
        double probForPredictedLabels = Double.NaN;

        if (scaling instanceof MultiLabelClassifier.AssignmentProbEstimator){
            probForPredictedLabels = ((MultiLabelClassifier.AssignmentProbEstimator) scaling).predictAssignmentProb(dataSet.getRow(dataPointIndex),
                    predictedLabels);
        }

        predictionAnalysis.setProbForPredictedLabels(probForPredictedLabels);

        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k: classes){
            ClassScoreCalculation classScoreCalculation = decisionProcess(boosting,scaling,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,limit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);

        List<MultiLabelPredictionAnalysis.ClassRankInfo> ranking = classes.stream().map(label -> {
                    MultiLabelPredictionAnalysis.ClassRankInfo rankInfo = new MultiLabelPredictionAnalysis.ClassRankInfo();
                    rankInfo.setClassIndex(label);
                    rankInfo.setClassName(labelTranslator.toExtLabel(label));
                    rankInfo.setProb(scaling.predictClassProb(dataSet.getRow(dataPointIndex), label));
                    return rankInfo;
                }
        ).collect(Collectors.toList());



        return predictionAnalysis;
    }




}
