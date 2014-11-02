package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Optional;

/**
 * Created by chengli on 8/5/14.
 */
public class BinarySplitter {

    //todo optimize, there is no need to use SSE
    static Optional<SplitResult> split(RegTreeConfig regTreeConfig,
                                       DataSet dataSet,
                                       double[] labels,
                                       double[] probs,
                                       int featureIndex){
        int numDataPoints = dataSet.getNumDataPoints();
        Vector featureValues;
        Vector inputVector = dataSet.getColumn(featureIndex);
        if (inputVector.isDense()){
            featureValues = inputVector;
        } else {
            featureValues = new DenseVector(inputVector);
        }

        System.out.println(featureValues);
        double parentSSE = ProbabilisticSSE.sse(labels,probs);

        //probabilistic counts
        double leftCountExisting = 0;
        double rightCountExisting = 0;
        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            if (featureValue==0){
                leftCountExisting += probs[i];
            }

            if (featureValue==1){
                rightCountExisting += probs[i];
            }

            //ignore NaN
        }

        double sumOfCounts = leftCountExisting + rightCountExisting;

        // all values are missing
        if (sumOfCounts==0){
            return Optional.empty();
        }

        // the prior probability of going left, decided based on all existing values
        double leftPriorProb = leftCountExisting/sumOfCounts;
        // the prior probability of going right, decided based on all existing values
        double rightPriorProb = rightCountExisting/sumOfCounts;

        double[] leftProbs = new double[numDataPoints];
        double[] rightProbs = new double[numDataPoints];

        for (int i=0;i<numDataPoints;i++){
            double featureValue = featureValues.get(i);
            // for missing values, just use prior probabilities
            if (Double.isNaN(featureValue)){
                leftProbs[i] = probs[i]*leftPriorProb;
                rightProbs[i] = probs[i]*rightPriorProb;
            }

            // for existing values, the partition is deterministic
            // we only need to keep the parent probability
            if (featureValue==0){
                leftProbs[i] = probs[i];
                rightProbs[i] = 0;
            }

            if (featureValue==1){
                leftProbs[i] = 0;
                rightProbs[i] = probs[i];
            }
        }

        System.out.println("left probs:");
        System.out.println(Arrays.toString(leftProbs));
        System.out.println("right probs:");
        System.out.println(Arrays.toString(rightProbs));

        //probabilistic count, including missing values
        double totalLeftCount = MathUtil.arraySum(leftProbs);
        double totalRightCount = MathUtil.arraySum(rightProbs);
        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
        boolean valid = (totalLeftCount>=minDataPerLeaf)&&(totalRightCount>=minDataPerLeaf);
        if (!valid){
            // no need to calculate reduction
            return Optional.empty();
        }

        double leftSSE = ProbabilisticSSE.sse(labels,leftProbs);
        double rightSSE = ProbabilisticSSE.sse(labels,rightProbs);
        double reduction = parentSSE - leftSSE - rightSSE;
        SplitResult splitResult = new SplitResult();
        splitResult.setFeatureIndex(featureIndex)
                .setThreshold(0).setReduction(reduction)
                .setLeftCount(totalLeftCount).setRightCount(totalRightCount);
        return Optional.of(splitResult);
    }


}
