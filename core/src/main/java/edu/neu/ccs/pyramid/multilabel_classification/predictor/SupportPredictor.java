package edu.neu.ccs.pyramid.multilabel_classification.predictor;

import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;

public class SupportPredictor implements PluginPredictor<MultiLabelClassifier.ClassProbEstimator> {

    private static final long serialVersionUID = 2L;

    MultiLabelClassifier.ClassProbEstimator classifier;
    LabelCalibrator labelCalibrator;
    VectorCalibrator setCalibrator;
    List<MultiLabel> support;
    PredictionFeatureExtractor predictionFeatureExtractor;

    public List<MultiLabel> getSupport() {
        return support;
    }

    public SupportPredictor(ClassProbEstimator classifier, LabelCalibrator labelCalibrator,
                            VectorCalibrator setCalibrator, PredictionFeatureExtractor predictionFeatureExtractor, List<MultiLabel> support) {
        this.classifier = classifier;
        this.labelCalibrator = labelCalibrator;
        this.setCalibrator = setCalibrator;
        this.predictionFeatureExtractor = predictionFeatureExtractor;
        this.support = support;
    }


//    public static MultiLabel predict(double[] marginals, List<MultiLabel> support){
//
//        return support.stream().map(m->new Pair<>(m, prob(marginals, m))).max(Comparator.comparing(Pair::getSecond))
//                .get().getFirst();
//    }


//    public static List<Pair<MultiLabel,Double>> topKSetsAndProbs(double[] marginals, List<MultiLabel> support, int top){
//        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(Pair::getSecond);
//        return support.stream().map(m->new Pair<>(m, prob(marginals, m))).sorted(comparator.reversed())
//                .limit(top).collect(Collectors.toList());
//    }

//    private static double prob(double[] marginals, MultiLabel multiLabel){
//        double p = 1;
//        for (int l=0;l<marginals.length;l++){
//            if (multiLabel.matchClass(l)){
//                p*= marginals[l];
//            } else {
//                p*= (1-marginals[l]);
//            }
//        }
//        return p;
//    }

    @Override
    public ClassProbEstimator getModel() {
        return classifier;
    }


    //todo make it better
    // should dp be used here?
    // should candidate number be a parameter?
    @Override
    public MultiLabel predict(Vector vector) {
        double[] uncali = classifier.predictClassProbs(vector);
        double[] marginals = labelCalibrator.calibratedClassProbs(uncali);
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();

        Set<MultiLabel> supportSet = new HashSet<>(support);
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        for (int i=0;i<=50;i++){
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (supportSet.contains(candidate)){
                PredictionCandidate predictionCandidate = new PredictionCandidate();
                predictionCandidate.x = vector;
                predictionCandidate.labelProbs = marginals;
                predictionCandidate.multiLabel = candidate;

                Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
                double score = setCalibrator.calibrate(feature);
                candidates.add(new Pair<>(candidate,score));
            }
        }

        if (candidates.isEmpty()){
            for (MultiLabel candidate: support){
                PredictionCandidate predictionCandidate = new PredictionCandidate();
                predictionCandidate.x = vector;
                predictionCandidate.labelProbs = marginals;
                predictionCandidate.multiLabel = candidate;

                Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
                double score = setCalibrator.calibrate(feature);
                candidates.add(new Pair<>(candidate,score));
            }
        }



        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return candidates.stream().max(comparator).map(Pair::getFirst).get();
    }



//    public Pair<MultiLabel,Double> predictWithConfidence(Vector vector) {
//        double[] uncali = classifier.predictClassProbs(vector);
//        double[] cali = labelCalibrator.calibratedClassProbs(uncali);
//        return support.stream().map(m->new Pair<>(m, prob(cali, m))).max(Comparator.comparing(Pair::getSecond))
//                .get();
//    }
}
