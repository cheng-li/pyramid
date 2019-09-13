package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;


/**
 * find top K sets by calibrated probability
 */
public class TopKFinder {

    public static List<Pair<MultiLabel,Double>> topK(Vector x, MultiLabelClassifier.ClassProbEstimator classProbEstimator, LabelCalibrator labelCalibrator,
                                                     VectorCalibrator vectorCalibrator, PredictionFeatureExtractor predictionFeatureExtractor,
                                                     int top){
        List<Pair<MultiLabel,Double>> list = new ArrayList<>();
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(x));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);

        for (int i=0;i<top;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = x;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double pro = vectorCalibrator.calibrate(feature);
            list.add(new Pair<>(candidate,pro));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }


    public static List<Pair<MultiLabel,Double>> topK(Vector x, MultiLabelClassifier.ClassProbEstimator classProbEstimator, LabelCalibrator labelCalibrator,
                                                     VectorCalibrator vectorCalibrator, PredictionFeatureExtractor predictionFeatureExtractor,
                                                     int minSetSize, int maxSetSize,
                                                     int top){
        List<Pair<MultiLabel,Double>> list = new ArrayList<>();
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(x));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);

        List<MultiLabel> candidates = new ArrayList<>();
        for (int i=0;i<top;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (candidate.getNumMatchedLabels()>=minSetSize&&candidate.getNumMatchedLabels()<=maxSetSize){
                candidates.add(candidate);
            }
        }

        if (candidates.isEmpty()){
            int[] sorted = ArgSort.argSortDescending(marginals);
            MultiLabel candidate = new MultiLabel();
            for (int i=0;i<minSetSize;i++){
                candidate.addLabel(sorted[i]);
            }
            candidates.add(candidate);
        }

        for (MultiLabel candidate: candidates){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = x;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double pro = vectorCalibrator.calibrate(feature);
            list.add(new Pair<>(candidate,pro));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }

    public static List<Pair<MultiLabel,Double>> topKinSupport(Vector x, MultiLabelClassifier.ClassProbEstimator classProbEstimator, LabelCalibrator labelCalibrator,
                                                              VectorCalibrator vectorCalibrator, PredictionFeatureExtractor predictionFeatureExtractor,
                                                              List<MultiLabel> support,
                                                              int top){
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(x));
        Set<MultiLabel> supportSet = new HashSet<>(support);
        List<Pair<MultiLabel,Double>> list = new ArrayList<>();
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<MultiLabel> candidates = new ArrayList<>();
        for (int i=0;i<top;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (supportSet.contains(candidate)){
                candidates.add(candidate);
            }
        }

        for (MultiLabel candidate: candidates){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = x;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double pro = vectorCalibrator.calibrate(feature);
            list.add(new Pair<>(candidate,pro));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }


}
