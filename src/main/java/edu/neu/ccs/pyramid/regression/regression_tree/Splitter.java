package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/6/14.
 */
public class Splitter {
    private static final Logger logger = LogManager.getLogger();
    private static ForkJoinPool pool = new ForkJoinPool();


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
        if (logger.isDebugEnabled()){
            logger.debug("global statistics = "+globalStats);
        }

        int randomLevel = regTreeConfig.getRandomLevel();

        ForkJoinTask<List<SplitResult>> task = pool.submit(() ->
                IntStream.range(0, dataSet.getNumFeatures())
                        .parallel()
                        .mapToObj(featureIndex -> split(regTreeConfig, dataSet, labels,
                                probs, featureIndex, globalStats))
                        .filter(Optional::isPresent)
                        .map(Optional::get)
                        .sorted(Comparator.comparing(SplitResult::getReduction).reversed())
                        .limit(randomLevel)
                        .collect(Collectors.toList()));
        // the list might be empty
        List<SplitResult> splitResults = task.join();
        return sample(splitResults);
//
//        Optional<SplitResult> result = Arrays.stream(activeFeatures).parallel()
//                .mapToObj(featureIndex -> split(regTreeConfig,dataSet,labels,
//                        probs,featureIndex,globalStats))
//                .filter(Optional::isPresent)
//                .map(Optional::get)
//                .max(Comparator.comparing(SplitResult::getReduction));

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

        return IntStream.range(0,dataSet.getNumFeatures()).parallel()
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

    static Optional<SplitResult> sample(List<SplitResult> splitResults){
        if (splitResults.size()==0){
            return Optional.empty();
        }

        if (splitResults.get(0).getReduction()==0){
            return Optional.empty();
        }

        double total = splitResults.stream().mapToDouble(SplitResult::getReduction).sum();
        double[] probs = splitResults.stream().mapToDouble(splitResult -> splitResult.getReduction()/total)
                .toArray();
        int[] singletons = IntStream.range(0,splitResults.size()).toArray();
        EnumeratedIntegerDistribution distribution = new EnumeratedIntegerDistribution(singletons,probs);
        int sample = distribution.sample();
        return Optional.of(splitResults.get(sample));
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

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder("GlobalStats{");
            sb.append("WeightedLabelSum=").append(WeightedLabelSum);
            sb.append(", probabilisticCount=").append(probabilisticCount);
            sb.append(", binaryCount=").append(binaryCount);
            sb.append('}');
            return sb.toString();
        }
    }
}
