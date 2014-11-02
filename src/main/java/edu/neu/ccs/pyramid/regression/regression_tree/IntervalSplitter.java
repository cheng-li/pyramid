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
                                       int featureIndex){
        Vector featureValues;
        Vector inputVector = dataSet.getColumn(featureIndex).getVector();
        if (inputVector.isDense()){
            featureValues = inputVector;
        } else {
            featureValues = new DenseVector(inputVector);
        }



        List<Interval> possibleIntervals;
        if (dataSet.hasMissingValue()){
            possibleIntervals = generateIntervalsWithMissingValue(regTreeConfig, featureValues, probs, labels);
        } else {
            possibleIntervals = generateIntervalsWithoutMissingValue(regTreeConfig,featureValues,probs,labels);
        }

        List<Interval> compressedIntervals = compress(possibleIntervals);
        return findBest(regTreeConfig,compressedIntervals,featureIndex);
    }

    static List<Interval> generateIntervalsWithMissingValue(RegTreeConfig regTreeConfig,
                                                            Vector featureValues,
                                                            double[] probs,
                                                            double[] labels){
        int numDataPoints = featureValues.size();
        int numIntervals = regTreeConfig.getNumSplitIntervals();
        // find min and max
        // we cannot start with featureValues[0] as it may be NaN
        double maxFeature = Double.NEGATIVE_INFINITY;
        double minFeature = Double.POSITIVE_INFINITY;
        double existingProbCount = 0;
        int existingBinaryCount = 0;
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            // only estimate min and max with actually present values
            // as the tree grows, we want to focus on smaller regions
            // if we don't impose probs[i]!=0, we will always use the global min and max
            if (!Double.isNaN(featureValue) && probs[i]!=0){
                existingProbCount += probs[i];
                existingBinaryCount += 1;
                if (featureValue > maxFeature){
                    maxFeature = featureValue;
                }
                if (featureValue < minFeature){
                    minFeature = featureValue;
                }
            }
        }

        List<Interval> intervals = new ArrayList<>(numIntervals);

        // if there is no more than 2 existing values, return empty intervals
        if (existingBinaryCount < 2){
            return intervals;
        }

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

        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            double label = labels[i];
            // if probs[i]==0, its feature value may be bigger than max or smaller than min,
            // so we should skip it
            if (!Double.isNaN(featureValue) && probs[i]!=0){
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
                Interval interval = intervals.get(intervalIndex);
                // conditional probability = 1 for matched interval
                double probability = probs[i];
                double oldCount = interval.getProbabilisticCount();
                interval.setProbabilisticCount(oldCount + probability);
                double oldSum = interval.getWeightedSum();
                interval.setWeightedSum(oldSum + label*probability);
            }
        }

        // estimate percentage for each interval
        for (Interval interval: intervals){
            interval.setPercentage(interval.getProbabilisticCount()/existingProbCount);
        }


        // second scan, only deal with missing values
        // assign missing values to every interval probabilistically
        // follow estimated percentages
        // conditional probability = percentage
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            double label = labels[i];
            if (Double.isNaN(featureValue)){
                for (Interval interval: intervals){
                    double probability = probs[i] * interval.getPercentage();
                    double oldCount = interval.getProbabilisticCount();
                    interval.setProbabilisticCount(oldCount + probability);
                    double oldSum = interval.getWeightedSum();
                    interval.setWeightedSum(oldSum + label*probability);
                }
            }
        }

        return intervals;
    }

    static List<Interval> generateIntervalsWithoutMissingValue(RegTreeConfig regTreeConfig,
                                                            Vector featureValues,
                                                            double[] probs,
                                                            double[] labels){
        int numDataPoints = featureValues.size();
        int numIntervals = regTreeConfig.getNumSplitIntervals();
        // find min and max
        double maxFeature = Double.NEGATIVE_INFINITY;
        double minFeature = Double.POSITIVE_INFINITY;
        double existingProbCount = 0;
        int existingBinaryCount = 0;
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            // only estimate min and max with actually present values
            // as the tree grows, we want to focus on smaller regions
            // if we don't impose probs[i]!=0, we will always use the global min and max
            if (probs[i]!=0){
                existingProbCount += probs[i];
                existingBinaryCount += 1;
                if (featureValue > maxFeature){
                    maxFeature = featureValue;
                }
                if (featureValue < minFeature){
                    minFeature = featureValue;
                }
            }
        }

        List<Interval> intervals = new ArrayList<>(numIntervals);

        // if there is no more than 2 existing values, return empty intervals
        if (existingBinaryCount < 2){
            return intervals;
        }

        double intervalLength = (maxFeature-minFeature)/numIntervals;

        for (int i=0;i<numIntervals;i++){
            Interval interval = new Interval();
            double lower = minFeature + i*intervalLength;
            double upper = lower + intervalLength;
            interval.setLower(lower);
            interval.setUpper(upper);
            intervals.add(interval);
        }

        // assign each existing value to one interval deterministically
        // conditional probability = 1 for matched interval and 0 for other intervals

        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            double label = labels[i];
            // if probs[i]==0, its feature value may be bigger than max or smaller than min,
            // so we should skip it
            if (probs[i]!=0){
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
                Interval interval = intervals.get(intervalIndex);
                // conditional probability = 1 for matched interval
                double probability = probs[i];
                double oldCount = interval.getProbabilisticCount();
                interval.setProbabilisticCount(oldCount + probability);
                double oldSum = interval.getWeightedSum();
                interval.setWeightedSum(oldSum + label*probability);
            }
        }

        // estimate percentage for each interval
        for (Interval interval: intervals){
            interval.setPercentage(interval.getProbabilisticCount()/existingProbCount);
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
}



