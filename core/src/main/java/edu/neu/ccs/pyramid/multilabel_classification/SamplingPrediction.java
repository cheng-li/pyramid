package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/16/16.
 */
public class SamplingPrediction {
    public static MultiLabel predict(double[] probabilities, List<MultiLabel> candidates){
        int[] s = IntStream.range(0, probabilities.length).toArray();
        EnumeratedIntegerDistribution distribution = new EnumeratedIntegerDistribution(s, probabilities);
        int i = distribution.sample();
        return candidates.get(i);
    }
}
