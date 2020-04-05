package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;

public class Reranker implements MultiLabelClassifier, VectorCalibrator {
    private static final long serialVersionUID = 2L;
    Regressor regressor;
    MultiLabelClassifier.ClassProbEstimator classProbEstimator;
    int numCandidate;
    private PredictionFeatureExtractor predictionFeatureExtractor;
    private LabelCalibrator labelCalibrator;
    private int minPredictionSize = 0;
    private int maxPredictionSize = Integer.MAX_VALUE;



    public Reranker(Regressor regressor, ClassProbEstimator classProbEstimator, int numCandidate,
                    PredictionFeatureExtractor predictionFeatureExtractor, LabelCalibrator labelCalibrator) {
        this.regressor = regressor;
        this.classProbEstimator = classProbEstimator;
        this.numCandidate = numCandidate;
        this.predictionFeatureExtractor = predictionFeatureExtractor;
        this.labelCalibrator = labelCalibrator;
    }

    public Regressor getRegressor() {
        return regressor;
    }


    public void setMinPredictionSize(int minPredictionSize) {
        this.minPredictionSize = minPredictionSize;
    }

    public void setMaxPredictionSize(int maxPredictionSize) {
        this.maxPredictionSize = maxPredictionSize;
    }

    @Override
    public int getNumClasses() {
        return classProbEstimator.getNumClasses();
    }

    public double prob(Vector vector, MultiLabel multiLabel){
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(vector));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<Pair<MultiLabel,Double>> topK = dynamicProgramming.topK(numCandidate);

        PredictionCandidate predictionCandidate = new PredictionCandidate();
        predictionCandidate.x = vector;
        predictionCandidate.labelProbs = marginals;
        predictionCandidate.multiLabel = multiLabel;
        predictionCandidate.sparseJoint = topK;
        Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
        double score = regressor.predict(feature);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }

    @Override
     public MultiLabel predict(Vector vector) {
        double[] uncalibratedMarginals = classProbEstimator.predictClassProbs(vector);
        return predict(vector, uncalibratedMarginals);
    }


    public MultiLabel predict(Vector vector, double[] uncalibratedLabelScores) {
        double[] marginals = labelCalibrator.calibratedClassProbs(uncalibratedLabelScores);

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<Pair<MultiLabel,Double>> sparseJoint = dynamicProgramming.topK(numCandidate);

        List<MultiLabel> multiLabels = sparseJoint.stream().map(pair->pair.getFirst())
                .filter(candidate->candidate.getNumMatchedLabels() >= minPredictionSize && candidate.getNumMatchedLabels() <= maxPredictionSize)
                .collect(Collectors.toList());
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();

        if (multiLabels.isEmpty()){
            int[] sorted = ArgSort.argSortDescending(marginals);
            MultiLabel multiLabel = new MultiLabel();
            for (int i=0;i<minPredictionSize;i++){
                multiLabel.addLabel(sorted[i]);
            }
            multiLabels.add(multiLabel);
        }

        for (MultiLabel candidate: multiLabels){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = vector;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;
            predictionCandidate.sparseJoint = sparseJoint;
            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = regressor.predict(feature);
            candidates.add(new Pair<>(candidate,score));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return candidates.stream().max(comparator).map(Pair::getFirst).get();
    }


//    public MultiLabel predictByGFM(Vector vector){
//        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(vector));
//        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
//        List<MultiLabel> multiLabels = new ArrayList<>();
//        List<Double> probabilities = new ArrayList<>();
//
//        for (int i=0;i<numCandidate;i++){
//            MultiLabel candidate = dynamicProgramming.nextHighestVector();
//
//            PredictionCandidate predictionCandidate = new PredictionCandidate();
//            predictionCandidate.x = vector;
//            predictionCandidate.labelProbs = marginals;
//            predictionCandidate.multiLabel = candidate;
//
//            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
//            double score = regressor.predict(feature);
//            multiLabels.add(candidate);
//            probabilities.add(score);
//        }
//        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
//        return generalF1Predictor.predict(classProbEstimator.getNumClasses(),multiLabels,probabilities);
//    }



//    public boolean isInTopK(Vector vector,  MultiLabel groundTruth){
//        double[] marginals = predictionVectorizer.getLabelCalibrator().calibratedClassProbs(classProbEstimator.predictClassProbs(vector));
//        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
//        for (int i=0;i<numCandidate;i++) {
//            MultiLabel candidate = dynamicProgramming.nextHighestVector();
//            if (candidate.equals(groundTruth)){
//                return true;
//            }
//        }
//        return false;
//    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    @Override
    public double calibrate(Vector vector) {
        double score = regressor.predict(vector);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Reranker{");
        sb.append("regressor=").append(regressor);
        sb.append('}');
        return sb.toString();
    }
}
