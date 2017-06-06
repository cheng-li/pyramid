package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.util.ArgSort;

import java.util.stream.IntStream;

/**
 * Created by chengli on 8/20/14.
 */
public class AppearanceAtTop {

    public static double rate(Classifier.ProbabilityEstimator probabilityEstimator,
                              ClfDataSet clfDataSet,
                              int top){
        int[] labels = clfDataSet.getLabels();
        long total = IntStream.range(0, clfDataSet.getNumDataPoints()).parallel()
                .filter(i -> appear(labels[i],
                        probabilityEstimator.predictClassProbs(clfDataSet.getRow(i)),
                        top))
                .count();
        return (double)total/clfDataSet.getNumDataPoints();
    }

    /**
     *
     * @param label
     * @param measures based on which rank is calculated, can be scores or probabilities,
     *                 descending order
     * @param top
     * @return
     */
    static boolean appear(int label, double[] measures, int top){
        int[] rankedIndices = ArgSort.argSortDescending(measures);
        return appear(label,rankedIndices,top);
    }

    static boolean appear(int label, int[] rankedIndices, int top){
        if (top<1){
            throw new IllegalArgumentException("top must be at least 1");
        }
        for (int i=0;i<top;i++){
            if (rankedIndices[i]==label){
                return true;
            }
        }
        return false;
    }
}
