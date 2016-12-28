package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.List;
import java.util.stream.IntStream;

/**
 * mean average precision
 * Created by chengli on 12/27/16.
 */
public class MAP {
    /**
     * compute mean average precision over given labels
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double map(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet, List<Integer> labels){
        if (classifier.getNumClasses()!=dataSet.getNumClasses()){
            throw new IllegalArgumentException("classifier.getNumClasses()!=dataSet.getNumClasses()");
        }
        int numData = dataSet.getNumDataPoints();
        double[][] probs = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];

        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->probs[i] = classifier.predictClassProbs(dataSet.getRow(i)));

        double sum = 0;

        for (int l: labels){
            int[] binaryLabels = new int[numData];
            double[] marginals = new double[numData];
            for (int i=0;i<numData;i++){
                if (dataSet.getMultiLabels()[i].matchClass(l)){
                    binaryLabels[i] = 1;
                }
                marginals[i] = probs[i][l];
            }

            double averagePrecision = AveragePrecision.averagePrecision(binaryLabels, marginals);
            sum += averagePrecision;
        }
        return sum/labels.size();
    }
}
