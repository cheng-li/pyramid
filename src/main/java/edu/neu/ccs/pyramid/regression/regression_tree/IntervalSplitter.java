package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Created by chengli on 8/6/14.
 */
class IntervalSplitter {

    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs,
                                       int featureIndex,
                                       Splitter.GlobalStats globalStats){
        Vector featureValues = dataSet.getColumn(featureIndex);
        List<Interval> possibleIntervals = generateIntervals(regTreeConfig, featureValues, probs, labels, globalStats);
        List<Interval> compressedIntervals = compress(possibleIntervals);
        return findBest(regTreeConfig,compressedIntervals,featureIndex);
    }

    static List<Interval> generateIntervals(RegTreeConfig regTreeConfig,
                                            Vector featureValues,
                                            double[] probs,
                                            double[] labels,
                                            Splitter.GlobalStats globalStats){
        FeatureStats featureStats = new FeatureStats(featureValues,probs,labels, globalStats);

        int numIntervals = regTreeConfig.getNumSplitIntervals();

        List<Interval> intervals = new ArrayList<>(numIntervals);

        double maxFeature = featureStats.getMax();
        double minFeature = featureStats.getMin();

        // if no present values, do nothing
        if (maxFeature == Double.NEGATIVE_INFINITY){
            return intervals;
        }

        if (minFeature == Double.POSITIVE_INFINITY){
            return intervals;
        }

        // now max and min should be finite
        //if no range, do nothing
        if (minFeature == maxFeature){
            return intervals;
        }

        //at this time, max and min should be finite numbers, and max > min
        // should generate intervals

        double intervalLength = (maxFeature-minFeature)/numIntervals;

        for (int i=0;i<numIntervals;i++){
            Interval interval = new Interval();
            double lower = minFeature + i*intervalLength;
            double upper = lower + intervalLength;
            interval.setLower(lower);
            interval.setUpper(upper);
            intervals.add(interval);
        }

        // first scan, only deal with existing values,
        // assign each existing value to one interval deterministically
        // conditional probability = 1 for matched interval and 0 for other intervals

        // for non-zero values
        if (featureStats.getNonZeroBinaryCount()>0){
            for (Vector.Element element: featureValues.nonZeroes()){
                int i = element.index();
                double featureValue = featureValues.get(i);
                double label = labels[i];
                // if probs[i]==0, its feature value may be bigger than max or smaller than min,
                // so we should skip it
                if (!Double.isNaN(featureValue) && probs[i]!=0){
                    int intervalIndex = getIntervalIndex(featureValue,minFeature,intervalLength,numIntervals);
                    Interval interval = intervals.get(intervalIndex);
                    // conditional probability = 1 for matched interval
                    double probability = probs[i];
                    double oldProbCount = interval.getProbabilisticCount();
                    interval.setProbabilisticCount(oldProbCount + probability);
                    double oldWeightedLabelSum = interval.getWeightedSum();
                    interval.setWeightedSum(oldWeightedLabelSum + label*probability);
                }
            }
        }


        // for zero values, do all of them together, as they all go to the same interval
        if (featureStats.getZeroBinaryCount()>0){
            int intervalIndex = getIntervalIndex(0,minFeature,intervalLength,numIntervals);
            Interval interval = intervals.get(intervalIndex);
            double oldProbCount = interval.getProbabilisticCount();
            interval.setProbabilisticCount(oldProbCount + featureStats.getZeroProbCount());
            double oldWeightedLabelSum = interval.getWeightedSum();
            interval.setWeightedSum(oldWeightedLabelSum + featureStats.getZeroWeightedLabelSum());
        }



        // estimate percentage for each interval
        for (Interval interval: intervals){
            interval.setPercentage(interval.getProbabilisticCount()/globalStats.getProbabilisticCount());
        }


        // second scan, only deal with missing values
        // assign missing values to every interval probabilistically
        // follow estimated percentages
        // conditional probability = percentage
        // can be processed together
        if (featureStats.getNanBinaryCount()>0){
            //todo verify
            for (Interval interval: intervals){
                double oldCount = interval.getProbabilisticCount();
                interval.setProbabilisticCount(oldCount + interval.getPercentage() * featureStats.getNanProbCount());
                double oldSum = interval.getWeightedSum();
                interval.setWeightedSum(oldSum + interval.getPercentage() * featureStats.getNanWeightedLabelSum());
            }


//            for (Vector.Element element: featureValues.nonZeroes()){
//                int i = element.index();
//                double featureValue = featureValues.get(i);
//                double label = labels[i];
//                if (Double.isNaN(featureValue)){
//                    for (Interval interval: intervals){
//                        double probability = probs[i] * interval.getPercentage();
//                        double oldCount = interval.getProbabilisticCount();
//                        interval.setProbabilisticCount(oldCount + probability);
//                        double oldSum = interval.getWeightedSum();
//                        interval.setWeightedSum(oldSum + label*probability);
//                    }
//                }
//            }
        }

        return intervals;
    }



    /**
     * only keep non-empty intervals, also adjust(extend) their boundaries
     * @param intervals
     * @return
     */
    static List<Interval> compress(List<Interval> intervals){
        //whether we are in a zero block
        boolean inBlock = false;
        int start = 0;
        int end = 0;

        for (int i=0;i<intervals.size();i++){
            if (intervals.get(i).getProbabilisticCount()==0){
                // enter block
                if (!inBlock){
                    inBlock=true;
                    start = i;
                    end = i;
                } else {
                    //extend block
                    end = i;
                }
            } else{
                if (inBlock){
                    // exit block
                    inBlock = false;
                    double mid = (intervals.get(start).getLower()
                            +intervals.get(end).getUpper())/2;
                    //the first and last intervals should contain min and max
                    //so they shouldn't be zero intervals
                    intervals.get(start-1).setUpper(mid);
                    intervals.get(end+1).setLower(mid);
                }
            }
        }
        List<Interval> compressed = new ArrayList<>(intervals.size());
        for (Interval interval: intervals){
            if (interval.getProbabilisticCount()!=0){
                compressed.add(interval);
            }
        }
        return compressed;
    }

    private static Optional<SplitResult> findBest(RegTreeConfig regTreeConfig,
                                        List<Interval> intervals,
                                        int featureIndex){
        List<SplitResult> splitResults = new ArrayList<>(intervals.size());
        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
        double totalSum=0;
        double totalCount=0;
        for (Interval interval: intervals){
            totalCount += interval.getProbabilisticCount();
            totalSum += interval.getWeightedSum();
        }


        double leftSum = 0;
        double leftCount = 0;
        for (int i=0;i<=intervals.size()-2;i++) {
            Interval interval = intervals.get(i);
            leftCount += interval.getProbabilisticCount();
            leftSum += interval.getWeightedSum();
            double rightSum = totalSum - leftSum;
            double rightCount = totalCount - leftCount;
            double reduction = leftSum * leftSum / leftCount +
                    rightSum * rightSum / rightCount
                    - totalSum * totalSum / totalCount;
            SplitResult splitResult = new SplitResult();
            splitResult.setFeatureIndex(featureIndex)
                    .setLeftCount(leftCount)
                    .setRightCount(rightCount)
                    .setReduction(reduction)
                    .setThreshold(interval.getUpper());
            splitResults.add(splitResult);
        }
        return splitResults.stream().filter(splitResult
                -> splitResult.getLeftCount() >= minDataPerLeaf
                && splitResult.getRightCount() >= minDataPerLeaf)
                .max(Comparator.comparing(SplitResult::getReduction));
    }


    static int getIntervalIndex(double featureValue, double minFeature, double intervalLength, int numIntervals){
        int ceil = (int)Math.ceil((featureValue-minFeature)/intervalLength);
        //this should not happen in theory
        //add this to handle round error
        if (ceil>numIntervals){
            ceil=numIntervals;
        }
        int intervalIndex;
        if (ceil==0){
            intervalIndex = 0;
        } else {
            intervalIndex = ceil-1;
        }
        return intervalIndex;
    }

    static class FeatureStats {
        // each feature value can be one of the following three types:
        // zero, non-zero, NaN

        private int zeroBinaryCount;
        // does not include NaN
        private int nonZeroBinaryCount;
        private int nanBinaryCount;

        private double zeroProbCount;
        private double nonZeroProbCount;
        private double nanProbCount;

        private double zeroWeightedLabelSum;
        private double nonZeroWeightedLabelSum;
        private double nanWeightedLabelSum;

        // min of present values
        private double min;
        // max of present values
        private double max;

        /**
         * gather statistics for one feature column by scanning only non-zero elements
         * @param featureValues
         * @param probs
         * @param globalStats
         */
        FeatureStats(Vector featureValues, double[] probs, double[] labels, Splitter.GlobalStats globalStats) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            // the iterator considers non zero and NaN
            for (Vector.Element element: featureValues.nonZeroes()){
                int index = element.index();
                double value = element.get();
                double prob = probs[index];
                double label = labels[index];
                if (prob>0){
                    if (Double.isNaN(value)){
                        nanBinaryCount += 1;
                        nanProbCount += prob;
                        nanWeightedLabelSum += prob*label;
                    } else {
                        nonZeroBinaryCount += 1;
                        nonZeroProbCount += prob;
                        nonZeroWeightedLabelSum += prob*label;
                        if (value < min){
                            min = value;
                        }
                        if (value > max){
                            max = value;
                        }
                    }
                }
            }

            zeroBinaryCount = globalStats.getBinaryCount() - nonZeroBinaryCount - nanBinaryCount;
            zeroProbCount = globalStats.getProbabilisticCount() - nonZeroProbCount - nanProbCount;
            zeroWeightedLabelSum = globalStats.getWeightedLabelSum() - nonZeroWeightedLabelSum - nanWeightedLabelSum;

            if (min>0 && zeroBinaryCount >0){
                min = 0;
            }

            if (max<0 && zeroBinaryCount >0){
                max = 0;
            }

            // at this point, min /max must be a finite number unless every feature value is NaN
        }

        int getZeroBinaryCount() {
            return zeroBinaryCount;
        }

        int getNonZeroBinaryCount() {
            return nonZeroBinaryCount;
        }

        int getNanBinaryCount() {
            return nanBinaryCount;
        }

        double getMin() {
            return min;
        }

        double getMax() {
            return max;
        }

        public double getZeroProbCount() {
            return zeroProbCount;
        }

        public double getNonZeroProbCount() {
            return nonZeroProbCount;
        }

        public double getNanProbCount() {
            return nanProbCount;
        }

        public double getZeroWeightedLabelSum() {
            return zeroWeightedLabelSum;
        }

        public double getNonZeroWeightedLabelSum() {
            return nonZeroWeightedLabelSum;
        }

        public double getNanWeightedLabelSum() {
            return nanWeightedLabelSum;
        }
    }


}



