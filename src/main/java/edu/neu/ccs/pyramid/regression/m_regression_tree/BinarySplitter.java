package edu.neu.ccs.pyramid.regression.m_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Optional;

/**
 * Created by chengli on 8/5/14.
 */
public class BinarySplitter {

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
        for (int dataIndex: dataAppearance){
            partialFeatures[pos] = featureValues.get(dataIndex);
            partialLabels[pos] = labels[dataIndex];
            pos += 1;
        }
        return splitPartial(regTreeConfig,partialFeatures,partialLabels,featureIndex);
    }

    private static Optional<SplitResult> splitPartial(RegTreeConfig regTreeConfig,
                                     double[] featureValues,
                                     double[] labels,
                                     int featureIndex){
        int numZeros = 0;
        int numOnes = 0;
        double zeroLabelSum = 0;
        double oneLabelSum = 0;
        for (int i=0;i<featureValues.length;i++){
            double featureValue = featureValues[i];
            double label = labels[i];
            if (featureValue==0){
                numZeros += 1;
                zeroLabelSum += label;
            } else {
                numOnes += 1;
                oneLabelSum += label;
            }
        }
        double totalLabelSum = zeroLabelSum + oneLabelSum;
        double reduction = (zeroLabelSum * zeroLabelSum / numZeros) +
                (oneLabelSum * oneLabelSum / numOnes) -
                (totalLabelSum * totalLabelSum / featureValues.length);
        SplitResult splitResult = new SplitResult();
        splitResult.setFeatureIndex(featureIndex)
                .setThreshold(0).setReduction(reduction)
                .setLeftCount(numZeros).setRightCount(numOnes);
        int minDataPerLeaf = regTreeConfig.getMinDataPerLeaf();
        boolean valid = (numOnes >= minDataPerLeaf)&&(numZeros >= minDataPerLeaf);
        if (valid){
            return Optional.of(splitResult);
        } else {
            return Optional.empty();
        }

    }
}
