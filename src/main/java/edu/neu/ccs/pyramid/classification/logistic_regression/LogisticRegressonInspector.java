package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/**
 * Created by chengli on 12/7/14.
 */
public class LogisticRegressonInspector {
    public static List<Pair<Integer,String>> topFeatures(LogisticRegression logisticRegression,
                                                         int k){
        String[] featureNames = logisticRegression.getFeatureNames();
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,weights.size()).mapToObj(i -> new Pair<>(i,weights.get(i)))
                .filter(pair -> pair.getSecond()>0)
                .sorted(comparator.reversed())
                .map(Pair::getFirst)
                .map(i -> new Pair<>(i, featureNames[i]))
                .collect(Collectors.toList());
    }

    public static List<Pair<Integer,String>> topFeatures(LogisticRegression logisticRegression,
                                                         int k,
                                                         int limit){
        String[] featureNames = logisticRegression.getFeatureNames();
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return IntStream.range(0,weights.size()).mapToObj(i -> new Pair<>(i,weights.get(i)))
                .filter(pair -> pair.getSecond()>0)
                .sorted(comparator.reversed())
                .map(Pair::getFirst)
                .map(i -> new Pair<>(i, featureNames[i]))
                .limit(limit)
                .collect(Collectors.toList());
    }
}
