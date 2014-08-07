package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/6/14.
 */
public class IntervalSplitter {
    static SplitResult split(RegTreeConfig regTreeConfig,
                             int[] dataAppearance,
                             int featureIndex){
        DataSet dataSet = regTreeConfig.getDataSet();
        double[] labels = regTreeConfig.getLabels();
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
        for (int dataIndex: dataAppearance){
            partialFeatures[pos] = featureValues.get(dataIndex);
            partialLabels[pos] = labels[dataIndex];
            pos += 1;
        }
        return splitPartial(regTreeConfig,partialFeatures,partialLabels,featureIndex);
    }

    private static SplitResult splitPartial(RegTreeConfig regTreeConfig,
                                            double[] featureValues,
                                            double[] labels,
                                            int featureIndex){
        int numDataPoints = featureValues.length;
        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
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

        //generate statistics
        double labelSum=0;
        double intervalLength = (maxFeature-minFeature)/numIntervals;
        //order: from small value to big value
        //counts and sum of labels in each interval
        int[] intervalCounts = new int[numIntervals];
        int[] intervalLabelSums = new int[numIntervals];
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues[i];
            double label = labels[i];
            labelSum += label;
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
            intervalCounts[intervalIndex] += 1;
            intervalLabelSums[intervalIndex] += label;
        }

        //find best
        double maxlrNormalizedSquareSum = 0;
        double bestThreshold = 0;
        double leftSum = 0;
        int leftCount = 0;
        boolean existValid = false;
        for (int i=1;i<=numIntervals-1;i++){
            leftCount += intervalCounts[i-1];
            leftSum += intervalLabelSums[i-1];
            double rightSum = labelSum - leftSum;
            int rightCount = numDataPoints - leftCount;
            boolean valid = (leftCount>=minDataPerLeaf)&&(rightCount>=minDataPerLeaf);
            if (valid){
                existValid = true;
                double lrNormalizedSquareSum = leftSum*leftSum/leftCount +
                        rightSum*rightSum/rightCount;
                boolean update = false;
                if(lrNormalizedSquareSum > maxlrNormalizedSquareSum){
                    update = true;
                } else if(lrNormalizedSquareSum == maxlrNormalizedSquareSum){
                    // for equally good threshold, we prefer the one close to the middle
                    double distance = Math.abs(minFeature + i*intervalLength - (minFeature+maxFeature)/2.0);
                    double lastDistance = Math.abs(bestThreshold - (minFeature+maxFeature)/2.0);
                    if (distance < lastDistance){
//                        update=true;
                    }
                }

                if (update){
                    maxlrNormalizedSquareSum = lrNormalizedSquareSum;
                    bestThreshold = minFeature + i*intervalLength;
                }
            }
        }
        SplitResult splitResult;
        if (existValid){
            double reduction = maxlrNormalizedSquareSum - labelSum*labelSum/numDataPoints;
            splitResult = new SplitResult(featureIndex,bestThreshold,reduction);
        } else {
            splitResult = new SplitResult(featureIndex,0,0);
            splitResult.setValid(false);
        }
        return splitResult;
    }
}
