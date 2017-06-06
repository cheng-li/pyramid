package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.util.ArgSort;

import java.util.stream.IntStream;

/**
 * Created by chengli on 9/5/14.
 */
public class MRR {

    public static double mrr(Classifier.ProbabilityEstimator probabilityEstimator,
                             ClfDataSet clfDataSet){
        int[] labels = clfDataSet.getLabels();
        return IntStream.range(0, clfDataSet.getNumDataPoints()).parallel()
                .mapToDouble(i-> reciprocalRank(labels[i],
                        probabilityEstimator.predictClassProbs(clfDataSet.getRow(i))))
                .average().getAsDouble();
    }

    /**
     *
     * @param label
     * @param measures based on which rank is calculated, can be scores or probabilities,
     *                 descending order
     * @return
     */
    static double reciprocalRank(int label, double[] measures){
        int[] rankedIndices = ArgSort.argSortDescending(measures);
        return reciprocalRank(label,rankedIndices);
    }

    static double reciprocalRank(int label, int[] rankedIndices){
        int i = 0;
        while(label!=rankedIndices[i]){
            i += 1;
        }
        return 1.0/(i+1);
    }
}
