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
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;

public class Reranker implements MultiLabelClassifier, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    Regressor regressor;
    MultiLabelClassifier.ClassProbEstimator classProbEstimator;
    int numCandidate;
    private PredictionFeatureExtractor predictionFeatureExtractor;
    private LabelCalibrator labelCalibrator;


    public Reranker(Regressor regressor, ClassProbEstimator classProbEstimator, int numCandidate,
                    PredictionFeatureExtractor predictionFeatureExtractor, LabelCalibrator labelCalibrator) {
        this.regressor = regressor;
        this.classProbEstimator = classProbEstimator;
        this.numCandidate = numCandidate;
        this.predictionFeatureExtractor = predictionFeatureExtractor;
        this.labelCalibrator = labelCalibrator;
    }



    @Override
    public int getNumClasses() {
        return classProbEstimator.getNumClasses();
    }

    public double prob(Vector vector, MultiLabel multiLabel){
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(vector));
        PredictionCandidate predictionCandidate = new PredictionCandidate();
        predictionCandidate.x = vector;
        predictionCandidate.labelProbs = marginals;
        predictionCandidate.multiLabel = multiLabel;
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
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(vector));

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();
        for (int i=0;i<numCandidate;i++){
            MultiLabel candidate = dynamicProgramming.nextHighestVector();

            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = vector;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = regressor.predict(feature);
            candidates.add(new Pair<>(candidate,score));
        }
        //todo
//        AccPredictor accPredictor = new AccPredictor(cbm);
//        accPredictor.setComponentContributionThreshold(0.001);
//        MultiLabel cbmPre = accPredictor.predict(vector);
//        Vector feature = predictionVectorizer.feature(bmDistribution, cbmPre,marginals,Optional.of(positionMap));
//        double score = regressor.predict(feature);
//        candidates.add(new Pair<>(cbmPre,score));


        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
//        double maxC = candidates.stream().max(comparator).map(pair->pair.getSecond()).get();
//        List<MultiLabel> ties = candidates.stream().filter(pair->pair.getSecond()==maxC).map(pair->pair.getFirst()).collect(Collectors.toList());
//        if (ties.size()>1){
//            System.out.println("marginals = "+ PrintUtil.printWithIndex(marginals));
//            System.out.println("number of ties = "+ties.size()+", "+ties);
//        }

        return candidates.stream().max(comparator).map(pair->pair.getFirst()).get();
    }


    public MultiLabel predict(Vector vector, double[] uncalibratedLabelScores) {
        double[] marginals = labelCalibrator.calibratedClassProbs(uncalibratedLabelScores);

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();
        for (int i=0;i<numCandidate;i++){
            MultiLabel candidate = dynamicProgramming.nextHighestVector();

            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = vector;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = regressor.predict(feature);
            candidates.add(new Pair<>(candidate,score));
        }
        //todo
//        AccPredictor accPredictor = new AccPredictor(cbm);
//        accPredictor.setComponentContributionThreshold(0.001);
//        MultiLabel cbmPre = accPredictor.predict(vector);
//        Vector feature = predictionVectorizer.feature(bmDistribution, cbmPre,marginals,Optional.of(positionMap));
//        double score = regressor.predict(feature);
//        candidates.add(new Pair<>(cbmPre,score));


        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
//        double maxC = candidates.stream().max(comparator).map(pair->pair.getSecond()).get();
//        List<MultiLabel> ties = candidates.stream().filter(pair->pair.getSecond()==maxC).map(pair->pair.getFirst()).collect(Collectors.toList());
//        if (ties.size()>1){
//            System.out.println("marginals = "+ PrintUtil.printWithIndex(marginals));
//            System.out.println("number of ties = "+ties.size()+", "+ties);
//        }

        return candidates.stream().max(comparator).map(pair->pair.getFirst()).get();
    }


    public MultiLabel predictByGFM(Vector vector){
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(vector));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<MultiLabel> multiLabels = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();

        for (int i=0;i<numCandidate;i++){
            MultiLabel candidate = dynamicProgramming.nextHighestVector();

            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = vector;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = regressor.predict(feature);
            multiLabels.add(candidate);
            probabilities.add(score);
        }
        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
        return generalF1Predictor.predict(classProbEstimator.getNumClasses(),multiLabels,probabilities);
    }



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
