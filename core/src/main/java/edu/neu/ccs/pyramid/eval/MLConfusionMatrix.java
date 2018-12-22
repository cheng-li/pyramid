package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * multi-label confusion matrix
 * use sparse matrix to avoid storing TN explicitly
 * convention: 0=TN, 1=TP, 2=FN, 3=FP
 * Created by chengli on 3/2/16.
 */
public class MLConfusionMatrix implements Serializable {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numDataPoints;
    //[numData][numClasses]
    private DataSet entries;

    public int getNumClasses() {
        return numClasses;
    }

    public DataSet getEntries() {
        return entries;
    }

    public int getNumDataPoints() {
        return numDataPoints;
    }

    public MLConfusionMatrix(int numClasses, MultiLabel[] trueLabels, MultiLabel[] predictions) {
        this.numClasses = numClasses;
        this.numDataPoints = trueLabels.length;
        int numData = trueLabels.length;
        this.entries = DataSetBuilder.getBuilder().numDataPoints(numDataPoints)
                .numFeatures(numClasses).density(Density.SPARSE_RANDOM).build();
        IntStream.range(0,numData).forEach(i->{
            MultiLabel label = trueLabels[i];
            MultiLabel prediction = predictions[i];
            Vector labelVector = label.toVector(numClasses);
            Vector predVector = prediction.toVector(numClasses);
            //todo speed up this by looking at non-zeros
            for (int l=0;l<numClasses;l++){
                double labelMatch = labelVector.get(l);
                double prediMatch = predVector.get(l);
                if (labelMatch==1&&prediMatch==1){
                    entries.setFeatureValue(i,l,1);
                } else if (labelMatch==1&&prediMatch==0){
                    entries.setFeatureValue(i,l,2);
                } else if (labelMatch==0&&prediMatch==0){
                    // do nothing
                } else {
                    entries.setFeatureValue(i,l,3);
                }
            }
        });
    }

    /**
     * assuming number of classes in dataset >= number of classes in classifier
     * @param classifier
     * @param dataSet
     */
    public MLConfusionMatrix(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        this(dataSet.getNumClasses(),dataSet.getMultiLabels(),classifier.predict(dataSet));
    }

    public MLConfusionMatrix(MultiLabelClfDataSet dataSet, MultiLabel[] predictions){
        this(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
    }


}
