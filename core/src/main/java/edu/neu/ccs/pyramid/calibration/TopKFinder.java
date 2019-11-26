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
        return topK(x, classProbEstimator, labelCalibrator, vectorCalibrator,predictionFeatureExtractor,0,Integer.MAX_VALUE,top);
    }


    public static List<Pair<MultiLabel,Double>> topK(Vector x, MultiLabelClassifier.ClassProbEstimator classProbEstimator, LabelCalibrator labelCalibrator,
                                                     VectorCalibrator vectorCalibrator, PredictionFeatureExtractor predictionFeatureExtractor,
                                                     int minSetSize, int maxSetSize,
                                                     int top){
        double[] uncalibratedMarginals = classProbEstimator.predictClassProbs(x);
        double[] marginals = labelCalibrator.calibratedClassProbs(uncalibratedMarginals);

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        //todo better
        List<Pair<MultiLabel,Double>> sparseJoint = dynamicProgramming.topK(50);

        List<MultiLabel> multiLabels = sparseJoint.stream().map(pair->pair.getFirst())
                .filter(candidate->candidate.getNumMatchedLabels() >= minSetSize && candidate.getNumMatchedLabels() <= maxSetSize)
                .collect(Collectors.toList());
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();

        if (multiLabels.isEmpty()){
            int[] sorted = ArgSort.argSortDescending(marginals);
            MultiLabel multiLabel = new MultiLabel();
            for (int i=0;i<minSetSize;i++){
                multiLabel.addLabel(sorted[i]);
            }
            multiLabels.add(multiLabel);
        }

        for (MultiLabel candidate: multiLabels){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = x;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;
            predictionCandidate.sparseJoint = sparseJoint;
            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = vectorCalibrator.calibrate(feature);
            candidates.add(new Pair<>(candidate,score));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return candidates.stream().sorted(comparator.reversed()).limit(top).collect(Collectors.toList());
    }

    public static List<Pair<MultiLabel,Double>> topKinSupport(Vector x, MultiLabelClassifier.ClassProbEstimator classProbEstimator, LabelCalibrator labelCalibrator,
                                                              VectorCalibrator vectorCalibrator, PredictionFeatureExtractor predictionFeatureExtractor,
                                                              List<MultiLabel> support,
                                                              int top){
        double[] marginals = labelCalibrator.calibratedClassProbs(classProbEstimator.predictClassProbs(x));


        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        //todo better
        List<Pair<MultiLabel,Double>> sparseJoint = dynamicProgramming.topK(50);

        List<Pair<MultiLabel,Double>> list = new ArrayList<>();

        for (MultiLabel candidate: support){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = x;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.multiLabel = candidate;
            predictionCandidate.sparseJoint = sparseJoint;

            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double pro = vectorCalibrator.calibrate(feature);
            list.add(new Pair<>(candidate,pro));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return list.stream().sorted(comparator.reversed()).limit(top).collect(Collectors.toList());
    }


}
