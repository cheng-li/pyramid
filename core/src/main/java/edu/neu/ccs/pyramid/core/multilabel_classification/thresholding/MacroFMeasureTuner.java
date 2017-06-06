package edu.neu.ccs.pyramid.core.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.core.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;

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

    private  static double tuneThreshold(MultiLabelClassifier.ClassProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet, int classIndex, double beta){
        double[] probs = new double[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            probs[i] = multiLabelClassifier.predictClassProb(dataSet.getRow(i),classIndex);
        }
        int[] labels = DataSetUtil.toBinaryLabels(dataSet.getMultiLabels(),classIndex);
        return BinaryFMeasureTuner.tuneThreshold(probs,labels,beta);
    }
}
