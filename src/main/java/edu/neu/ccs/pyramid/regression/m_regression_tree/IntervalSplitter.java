package edu.neu.ccs.pyramid.regression.m_regression_tree;

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
                                       int[] dataAppearance,
                                       int featureIndex){
        Vector featureValues;
        Vector inputVector = dataSet.getFeatureColumn(featureIndex).getVector();
        if (inputVector.isDense()){
            featureValues = inputVector;
        } else {
            featureValues = new DenseVector(inputVector);
        }
        int numDataPoints = dataAppearance.length;
        double[] partialLabels = new double[numDataPoints];
        double[] partialFeatures = new double[numDataPoints];
        int pos = 0;
        double labelSum = 0;
        for (int dataIndex: dataAppearance){
            partialFeatures[pos] = featureValues.get(dataIndex);
            double label = labels[dataIndex];
            partialLabels[pos] = label;
            labelSum += label;
            pos += 1;
        }
        List<Interval> possibleIntervals = generateIntervals(regTreeConfig,partialFeatures,partialLabels);
        List<Interval> compressedIntervals = compress(possibleIntervals);
        return findBest(regTreeConfig,compressedIntervals,
                labelSum,numDataPoints,featureIndex);
    }

    private static List<Interval> generateIntervals(RegTreeConfig regTreeConfig,
                                                    double[] featureValues,
                                                    double[] labels){
        int numDataPoints = featureValues.length;
        int numIntervals = regTreeConfig.getNumSplitIntervals();
        // find min and max
        double maxFeature = featureValues[0];
        double minFeature = featureValues[0];
        for (double featureValue: featureValues){
            if (featureValue > maxFeature){
                maxFeature = featureValue;
            }
            if (featureValue < minFeature){
                minFeature = featureValue;
            }
        }

        double intervalLength = (maxFeature-minFeature)/numIntervals;
        List<Interval> intervals = new ArrayList<>(numIntervals);
        for (int i=0;i<numIntervals;i++){
            Interval interval = new Interval();
            double lower = minFeature + i*intervalLength;
            double upper = lower + intervalLength;
            interval.setLower(lower);
            interval.setUpper(upper);
            intervals.add(interval);
        }
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues[i];
            double label = labels[i];

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
            interval.incrementCount();
            interval.incrementSum(label);
        }

        return intervals;
    }

    /**
     * only keep non-empty intervals, also adjust(extend) their boundaries
     * @param intervals
     * @return
     */
    private static List<Interval> compress(List<Interval> intervals){
        //whether we are in a zero block
        boolean inBlock = false;
        int start = 0;
        int end = 0;

        for (int i=0;i<intervals.size();i++){
            if (intervals.get(i).getCount()==0){
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
            if (interval.getCount()!=0){
                compressed.add(interval);
            }
        }
        return compressed;
    }

    private static Optional<SplitResult> findBest(RegTreeConfig regTreeConfig,
                                        List<Interval> intervals,
                                        double labelSum,
                                        int numDataPoints,
                                        int featureIndex){
        List<SplitResult> splitResults = new ArrayList<>(intervals.size());
        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();

        double leftSum = 0;
        int leftCount = 0;
        for (int i=0;i<=intervals.size()-2;i++) {
            Interval interval = intervals.get(i);
            leftCount += interval.getCount();
            leftSum += interval.getSum();
            double rightSum = labelSum - leftSum;
            int rightCount = numDataPoints - leftCount;
            double reduction = leftSum * leftSum / leftCount +
                    rightSum * rightSum / rightCount
                    - labelSum * labelSum / numDataPoints;
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


//==========================    old implementations  =======================

//    private static SplitResult findBest(RegTreeConfig regTreeConfig,
//                                        List<Interval> intervals,
//                                        double labelSum,
//                                        int numDataPoints,
//                                        int featureIndex){
//        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
//        double maxlrNormalizedSquareSum = -1;
//        double bestThreshold = 0;
//        double leftSum = 0;
//        int leftCount = 0;
//        boolean existValid = false;
//        for (int i=0;i<=intervals.size()-2;i++){
//            Interval interval = intervals.get(i);
//            leftCount += interval.getCount();
//            leftSum += interval.getSum();
//            double rightSum = labelSum - leftSum;
//            int rightCount = numDataPoints - leftCount;
//            boolean valid = (leftCount>=minDataPerLeaf)&&(rightCount>=minDataPerLeaf);
//            if (valid){
//                existValid = true;
//                double lrNormalizedSquareSum = leftSum*leftSum/leftCount +
//                        rightSum*rightSum/rightCount;
//                boolean update = false;
//                if(lrNormalizedSquareSum > maxlrNormalizedSquareSum){
//                    update = true;
//                } else if(lrNormalizedSquareSum == maxlrNormalizedSquareSum){
//                    // for equally good threshold, we flip a coin
//                    if (Math.random()<=0.5){
//                        update = true;
//                    }
//                }
//
//                if (update){
//                    maxlrNormalizedSquareSum = lrNormalizedSquareSum;
//                    bestThreshold = interval.getUpper();
//                }
//            }
//        }
//        SplitResult splitResult;
//        if (existValid){
//            double reduction = maxlrNormalizedSquareSum - labelSum*labelSum/numDataPoints;
//            splitResult = new SplitResult(featureIndex,bestThreshold,reduction);
//        } else {
//            splitResult = new SplitResult(featureIndex,0,0);
//            splitResult.setValid(false);
//        }
//        return splitResult;
//    }



//    private static SplitResult splitPartial(RegTreeConfig regTreeConfig,
//                                            double[] featureValues,
//                                            double[] labels,
//                                            int featureIndex){
//        int numDataPoints = featureValues.length;
//        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
//        int numIntervals = regTreeConfig.getNumSplitIntervals();
//
//        // find min and max
//        double maxFeature = featureValues[0];
//        double minFeature = featureValues[0];
//        for (double featureValue: featureValues){
//            if (featureValue > maxFeature){
//                maxFeature = featureValue;
//            }
//            if (featureValue < minFeature){
//                minFeature = featureValue;
//            }
//        }
//
//        //generate statistics
//        double labelSum=0;
//        double intervalLength = (maxFeature-minFeature)/numIntervals;
//        //order: from small value to big value
//        //counts and sum of labels in each interval
//        int[] intervalCounts = new int[numIntervals];
//        double[] intervalLabelSums = new double[numIntervals];
//        for (int i=0;i<numDataPoints;i++){
//            double featureValue = featureValues[i];
//            double label = labels[i];
//            labelSum += label;
//            int ceil = (int)Math.ceil((featureValue-minFeature)/intervalLength);
//            //this should not happen in theory
//            //add this to handle round error
//            if (ceil>numIntervals){
//                ceil=numIntervals;
//            }
//            int intervalIndex;
//            if (ceil==0){
//                intervalIndex = 0;
//            } else {
//                intervalIndex = ceil-1;
//            }
//            intervalCounts[intervalIndex] += 1;
//            intervalLabelSums[intervalIndex] += label;
//        }
////        System.out.println("threshods:");
////        for (int i=1;i<=numIntervals-1;i++){
////            System.out.println(minFeature+i*intervalLength);
////        }
////        System.out.println("label sum = "+labelSum);
////        System.out.println("interval counts "+ Arrays.toString(intervalCounts));
////        System.out.println("interval label sums "+Arrays.toString(intervalLabelSums));
//
//        //find best
//        double maxlrNormalizedSquareSum = 0;
//        double bestThreshold = 0;
//        double bestStartThreshold = 0;
//        double bestEndThreshold = 0;
//        double leftSum = 0;
//        int leftCount = 0;
//        boolean existValid = false;
//        for (int i=1;i<=numIntervals-1;i++){
//            leftCount += intervalCounts[i-1];
//            leftSum += intervalLabelSums[i-1];
//            double rightSum = labelSum - leftSum;
//            int rightCount = numDataPoints - leftCount;
//            boolean valid = (leftCount>=minDataPerLeaf)&&(rightCount>=minDataPerLeaf);
//            if (valid){
//                existValid = true;
//                double lrNormalizedSquareSum = leftSum*leftSum/leftCount +
//                        rightSum*rightSum/rightCount;
//                boolean update = false;
//                if(lrNormalizedSquareSum > maxlrNormalizedSquareSum){
//                    update = true;
//                } else if(lrNormalizedSquareSum == maxlrNormalizedSquareSum){
//                    // for equally good threshold, we prefer the one close to the middle
//                    double distance = Math.abs(minFeature + i*intervalLength - (minFeature+maxFeature)/2.0);
//                    double lastDistance = Math.abs(bestThreshold - (minFeature+maxFeature)/2.0);
//                    if (distance < lastDistance){
////                        update=true;
//                    }
//                }
//
//                if (update){
//                    maxlrNormalizedSquareSum = lrNormalizedSquareSum;
//                    bestThreshold = minFeature + i*intervalLength;
////                    System.out.println("best threshold ="+bestThreshold);
////                    System.out.println("left count "+leftCount);
////                    System.out.println("right count "+rightCount);
////                    System.out.println("left sum "+leftSum);
////                    System.out.println("right sum "+rightSum);
////                    System.out.println("maxlrNormalizedSquareSum "+maxlrNormalizedSquareSum);
//                }
//            }
//        }
//        SplitResult splitResult;
//        if (existValid){
//            double reduction = maxlrNormalizedSquareSum - labelSum*labelSum/numDataPoints;
//            splitResult = new SplitResult(featureIndex,bestThreshold,reduction);
//        } else {
//            splitResult = new SplitResult(featureIndex,0,0);
//            splitResult.setValid(false);
//        }
//        return splitResult;
//    }

