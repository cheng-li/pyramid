package edu.neu.ccs.pyramid.regression.regression_tree;

import com.google.common.util.concurrent.MoreExecutors;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureType;
import org.apache.commons.lang3.time.StopWatch;


import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/6/14.
 */
public class Splitter {
//    static  ExecutorService executor = MoreExecutors.getExitingExecutorService((ThreadPoolExecutor) Executors
//            .newFixedThreadPool(Runtime.getRuntime().availableProcessors()));


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
        Optional<SplitResult> result = Arrays.stream(activeFeatures).parallel()
                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
                        probs,featureIndex,globalStats))
                .filter(Optional::isPresent)
                .map(Optional::get)
                .max(Comparator.comparing(SplitResult::getReduction));
        return result;
    }


//    /**
//     *
//     * @param regTreeConfig
//     * @param probs
//     * @return best valid splitResult, possibly nothing
//     */
//    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
//                                       DataSet dataSet,
//                                       double[] labels,
//                                       double[] probs) {
//
//
//        GlobalStats globalStats = new GlobalStats(labels,probs);
//        int[] activeFeatures = regTreeConfig.getActiveFeatures();
//
//        List<Callable<Optional<SplitResult>>> tasks = new ArrayList<>(activeFeatures.length);
//
//        for (int i=0;i<activeFeatures.length;i++){
//            int featureIndex = activeFeatures[i];
//            tasks.add(()->split(regTreeConfig,dataSet,labels,probs,featureIndex,globalStats));
//        }
//
//        List<Future<Optional<SplitResult>>> futures = null;
//        try {
//            futures = executor.invokeAll(tasks);
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//
//        List<Optional<SplitResult>> optionals = new ArrayList<>(futures.size());
//        for (Future<Optional<SplitResult>> future: futures){
//            Optional<SplitResult> optional = null;
//            try {
//                optional = future.get();
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            } catch (ExecutionException e) {
//                e.printStackTrace();
//            }
//            optionals.add(optional);
//        }
//
//        Optional<SplitResult> result = optionals.parallelStream().filter(Optional::isPresent)
//                .map(Optional::get)
//                .max(Comparator.comparing(SplitResult::getReduction));
//
//        return result;
//    }


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
