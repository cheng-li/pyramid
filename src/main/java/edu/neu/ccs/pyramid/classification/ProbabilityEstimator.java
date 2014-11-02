package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public interface ProbabilityEstimator extends Classifier{
    double[] predictClassProbs(FeatureRow featureRow);

    default List<double[]> predictClassProbs(DataSet dataSet){
        return IntStream.range(0,dataSet.getNumDataPoints())
                .parallel().mapToObj(i -> predictClassProbs(dataSet.getRow(i)))
                .collect(Collectors.toList());
    }

    /**
     * by default, probabilities can be used for classification.
     * classifier should override this method if
     * calculation of probabilities is not necessary for classification, or
     * calculation of probabilities is too slow, or
     * calculation of probabilities is unstable
     * @param featureRow
     * @return
     */
    @Override
    default int predict(FeatureRow featureRow){
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        double[] probs = predictClassProbs(featureRow);
        return IntStream.range(0,probs.length)
                .mapToObj(i-> new Pair<>(i,probs[i]))
                .max(comparator).get().getFirst();
    }

}
