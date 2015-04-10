package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.classification.ClassProbability;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import edu.neu.ccs.pyramid.regression.ConstantRule;
import edu.neu.ccs.pyramid.regression.LinearRule;
import edu.neu.ccs.pyramid.regression.Rule;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 2/5/15.
 */
public class MLLogisticRegressionInspector {

    public static TopFeatures topFeatures(MLLogisticRegression logisticRegression,
                                                   int classIndex, int limit){
        FeatureList featureList = logisticRegression.getFeatureList();
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(classIndex);
        Comparator<FeatureUtility> comparator = Comparator.comparing(FeatureUtility::getUtility);
        List<Feature> list = IntStream.range(0, weights.size())
                .mapToObj(i -> new FeatureUtility(featureList.get(i)).setUtility(weights.get(i)))
                .filter(featureUtility -> featureUtility.getUtility()>0)
                .sorted(comparator.reversed())
                .map(FeatureUtility::getFeature)
                .limit(limit)
                .collect(Collectors.toList());
        TopFeatures topFeatures = new TopFeatures();
        topFeatures.setTopFeatures(list);
        topFeatures.setClassIndex(classIndex);
        LabelTranslator labelTranslator = logisticRegression.getLabelTranslator();
        topFeatures.setClassName(labelTranslator.toExtLabel(classIndex));
        return topFeatures;
    }

    public static ClassScoreCalculation decisionProcess(MLLogisticRegression logisticRegression,
                                                        LabelTranslator labelTranslator, Vector vector, int classIndex, int limit){
        ClassScoreCalculation classScoreCalculation = new ClassScoreCalculation(classIndex,labelTranslator.toExtLabel(classIndex),
                logisticRegression.predictClassScore(vector,classIndex));
        List<LinearRule> linearRules = new ArrayList<>();
        Rule bias = new ConstantRule(logisticRegression.getWeights().getBiasForClass(classIndex));
        classScoreCalculation.addRule(bias);
        for (int j=0;j<logisticRegression.getNumFeatures();j++){
            Feature feature = logisticRegression.getFeatureList().get(j);
            double weight = logisticRegression.getWeights().getWeightsWithoutBiasForClass(classIndex).get(j);
            double featureValue = vector.get(j);
            double score = weight*featureValue;
            LinearRule rule = new LinearRule();
            rule.setFeature(feature);
            rule.setFeatureValue(featureValue);
            rule.setScore(score);
            rule.setWeight(weight);
            linearRules.add(rule);
        }

        Comparator<LinearRule> comparator = Comparator.comparing(decision -> Math.abs(decision.getScore()));
        List<LinearRule> sorted = linearRules.stream().sorted(comparator.reversed()).limit(limit).collect(Collectors.toList());

        for (LinearRule linearRule : sorted){
            classScoreCalculation.addRule(linearRule);
        }

        return classScoreCalculation;
    }


    public static MultiLabelPredictionAnalysis analyzePrediction(MLLogisticRegression logisticRegression, MultiLabelClfDataSet dataSet,
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
        predictionAnalysis.setProbForTrueLabels(Double.NaN);

        MultiLabel predictedLabels = logisticRegression.predict(dataSet.getRow(dataPointIndex));
        List<Integer> internalPrediction = predictedLabels.getMatchedLabelsOrdered();
        predictionAnalysis.setInternalPrediction(internalPrediction);
        List<String> prediction = internalPrediction.stream().map(labelTranslator::toExtLabel).collect(Collectors.toList());
        predictionAnalysis.setPrediction(prediction);
        predictionAnalysis.setProbForPredictedLabels(Double.NaN);

        List<ClassScoreCalculation> classScoreCalculations = new ArrayList<>();
        for (int k: classes){
            ClassScoreCalculation classScoreCalculation = decisionProcess(logisticRegression,labelTranslator,
                    dataSet.getRow(dataPointIndex),k,limit);
            classScoreCalculations.add(classScoreCalculation);
        }
        predictionAnalysis.setClassScoreCalculations(classScoreCalculations);
        return predictionAnalysis;
    }
}
