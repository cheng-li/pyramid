package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.SerializableVector;
import org.apache.mahout.math.SequentialAccessSparseVector;

import java.io.IOException;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 8/31/17
 */
public class RowSparseSeqMLClfDataSet extends RowSparseSeqDataSet implements RowMultiLabelClfDataSet{
    private int numClasses;
    private MultiLabel[] multiLabels;
    private LabelTranslator labelTranslator;

    public RowSparseSeqMLClfDataSet(int numDatapoints, int numFeatures, int numClasses) {
        super(numDatapoints, numFeatures);
        this.numClasses = numClasses;
        this.multiLabels = new MultiLabel[numDatapoints];
        IntStream.range(0, numDatapoints).parallel().forEach(i -> {
                multiLabels[i] = new MultiLabel();
        });
//        for (int i=0; i<numDatapoints; i++) {
//            multiLabels[i] = new MultiLabel();
//        }
    }

    @Override
    public MultiLabel[] getMultiLabels() {
        return multiLabels;
    }

    @Override
    public void addLabel(int dataPointIndex, int classIndex) {
        multiLabels[dataPointIndex].addLabel(classIndex);
    }

    @Override
    public void setLabels(int dataPointIndex, MultiLabel multiLabel) {
        multiLabels[dataPointIndex] = multiLabel;
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }


    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        out.defaultWriteObject();
        SerializableVector[] serFeatureRows = new SerializableVector[featureRows.length];
        for (int i=0;i<featureRows.length;i++){
            serFeatureRows[i] = new SerializableVector(featureRows[i]);
        }
        out.writeObject(serFeatureRows);
    }

    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        in.defaultReadObject();
        SerializableVector[] serFeatureRows = (SerializableVector[])in.readObject();
        featureRows = new SequentialAccessSparseVector[serFeatureRows.length];
        for (int i=0;i<featureRows.length;i++){
            featureRows[i] = (SequentialAccessSparseVector) serFeatureRows[i].getVector();
        }
    }
}
