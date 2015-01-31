package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureType;


import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Created by chengli on 8/6/14.
 */
public class Splitter {
    /**
     *
     * @param regTreeConfig
     * @param probs
     * @return best valid splitResult, possibly nothing
     */
    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs){
        GlobalStats globalStats = new GlobalStats(labels,probs);
        int[] activeFeatures = regTreeConfig.getActiveFeatures();
        return Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        probs,featureIndex,globalStats))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .max(Comparator.comparing(SplitResult::getReduction));
    }


    public static List<SplitResult> getAllSplits(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs){
        GlobalStats globalStats = new GlobalStats(labels,probs);
        int[] activeFeatures = regTreeConfig.getActiveFeatures();
        return Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        probs,featureIndex, globalStats))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());
    }

    public static List<SplitResult> getAllSplits(RegTreeConfig regTreeConfig,
                                                 DataSet dataSet,
                                                 double[] labels){
        double[] probs = new double[labels.length];
        for (int i=0;i<labels.length;i++){
            probs[i] = 1;
        }
        return getAllSplits(regTreeConfig,dataSet,labels,probs);
    }

    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs,
                                       int featureIndex,
                                       GlobalStats globalStats){
        Optional<SplitResult> splitResult = IntervalSplitter.split(regTreeConfig,dataSet,labels,
                    probs,featureIndex, globalStats);

        return splitResult;
    }

    static class GlobalStats {
        //\sum _i p_i * y_i
        private double WeightedLabelSum;
        // \sum _i p_i
        private double probabilisticCount;
        // number of elements with non-zero probabilities
        private int binaryCount;

        GlobalStats(double[] labels,
                    double[] probs) {
            for (int i=0;i<labels.length;i++){
                double label = labels[i];
                double prob = probs[i];
                WeightedLabelSum += label*prob;
                probabilisticCount += prob;
                if (prob>0){
                    binaryCount += 1;
                }
            }
        }

        public double getWeightedLabelSum() {
            return WeightedLabelSum;
        }

        public double getProbabilisticCount() {
            return probabilisticCount;
        }

        public int getBinaryCount() {
            return binaryCount;
        }
    }
}
