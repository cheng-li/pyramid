package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * multi-label confusion matrix
 * Created by chengli on 3/2/16.
 */
public class MLConfusionMatrix {
    private int numClasses;
    private int numDataPoints;
    //[numData][numClasses]
    private Entry[][] entries;

    public int getNumClasses() {
        return numClasses;
    }

    public Entry[][] getEntries() {
        return entries;
    }

    public int getNumDataPoints() {
        return numDataPoints;
    }

    public MLConfusionMatrix(int numClasses, MultiLabel[] trueLabels, MultiLabel[] predictions) {
        this.numClasses = numClasses;
        this.numDataPoints = trueLabels.length;
        int numData = trueLabels.length;
        this.entries = new Entry[numData][numClasses];
        IntStream.range(0,numData).parallel().forEach(i->{
            MultiLabel label = trueLabels[i];
            MultiLabel prediction = predictions[i];
            Vector labelVector = label.toVector(numClasses);
            Vector predVector = prediction.toVector(numClasses);
            for (int l=0;l<numClasses;l++){
                double labelMatch = labelVector.get(l);
                double prediMatch = predVector.get(l);
                if (labelMatch==1&&prediMatch==1){
                    entries[i][l]=Entry.TP;
                } else if (labelMatch==1&&prediMatch==0){
                    entries[i][l]=Entry.FN;
                } else if (labelMatch==0&&prediMatch==0){
                    entries[i][l]=Entry.TN;
                } else {
                    entries[i][l]=Entry.FP;
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

    public enum Entry{
        TP, TN, FP, FN
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("MLConfusionMatrix{");
        sb.append("numClasses=").append(numClasses);
        sb.append(", entries=").append(Arrays.deepToString(entries));
        sb.append('}');
        return sb.toString();
    }
}
