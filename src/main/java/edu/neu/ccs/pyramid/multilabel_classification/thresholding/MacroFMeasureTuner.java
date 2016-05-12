package edu.neu.ccs.pyramid.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

/**
 * Created by chengli on 5/12/16.
 */
public class MacroFMeasureTuner {

    public static double[] tuneThresholds(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, double beta){
        double[] thresholds = new double[multiLabelClassifier.getNumClasses()];
        for (int k=0;k<multiLabelClassifier.getNumClasses();k++){
            thresholds[k] = tuneThreshold(multiLabelClassifier,dataSet,k,beta);
        }
        return thresholds;
    }

    private  static double tuneThreshold(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, int classIndex, double beta){
        double[] probs = new double[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            probs[i] = multiLabelClassifier.predictClassProb(dataSet.getRow(i),classIndex);
        }
        int[] labels = DataSetUtil.toBinaryLabels(dataSet.getMultiLabels(),classIndex);
        return BinaryFMeasureTuner.tuneThreshold(probs,labels,beta);
    }
}
