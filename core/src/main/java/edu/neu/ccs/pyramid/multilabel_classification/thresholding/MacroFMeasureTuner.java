package edu.neu.ccs.pyramid.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.stream.IntStream;

/**
 * Created by chengli on 5/12/16.
 */
public class MacroFMeasureTuner {

    public static double[] tuneThresholds(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, double beta){
        double[] thresholds = new double[multiLabelClassifier.getNumClasses()];
        IntStream.range(0, multiLabelClassifier.getNumClasses()).parallel()
                .forEach(k->thresholds[k] = tuneThreshold(multiLabelClassifier,dataSet,k,beta));
        return thresholds;
    }

    /**
     * find the optimal threshold for a class if we have more than 10 positive examples
     * otherwise, return the default threshold 0.5
     * @param multiLabelClassifier
     * @param dataSet
     * @param classIndex
     * @param beta
     * @return
     */
    private  static double tuneThreshold(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, int classIndex, double beta){
        int[] labels = DataSetUtil.toBinaryLabels(dataSet.getMultiLabels(),classIndex);
        if (MathUtil.arraySum(labels)<10){
            return 0.5;
        }
        double[] probs = new double[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            probs[i] = multiLabelClassifier.predictClassProb(dataSet.getRow(i),classIndex);
        }

        return BinaryFMeasureTuner.tuneThreshold(probs,labels,beta);
    }
}
