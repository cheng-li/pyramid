package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/5/17.
 */
public class LogLikelihood {

    public static double averageLogLikelihood(MultiLabelClassifier.AssignmentProbEstimator assignmentProbEstimator, MultiLabelClfDataSet dataSet, List<Integer> unobservedLabels){
        // Here we do not use approximation
        double[] logLikelihoods = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->assignmentProbEstimator.predictLogAssignmentProb(dataSet.getRow(i),dataSet.getMultiLabels()[i]))
                .toArray();

        double average = IntStream.range(0, dataSet.getNumDataPoints()).filter(i->!containsNovelClass(dataSet.getMultiLabels()[i],unobservedLabels))
                .mapToDouble(i->logLikelihoods[i]).average().getAsDouble();
        return average;
    }



    private static boolean containsNovelClass(MultiLabel multiLabel, List<Integer> novelLabels){
        for (int l:novelLabels){
            if (multiLabel.matchClass(l)){
                return true;
            }
        }
        return false;
    }
}
