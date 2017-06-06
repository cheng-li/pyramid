package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by Rainicy on 1/3/16.
 */
public class SequentialSparseMLClfDataSet extends SequentialSparseDataSet implements MultiLabelClfDataSet {
    private int numClasses;
    private MultiLabel[] multiLabels;
    private LabelTranslator labelTranslator;

    public SequentialSparseMLClfDataSet(int numDataPoints, int numFeatures,
                              boolean missingValue, int numClasses){
        super(numDataPoints, numFeatures, missingValue);
        this.numClasses=numClasses;
        this.multiLabels=new MultiLabel[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.multiLabels[i]= new MultiLabel();
        }
        this.labelTranslator = LabelTranslator.newDefaultLabelTranslator(numClasses);
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public MultiLabel[] getMultiLabels() {
        return this.multiLabels;
    }

    @Override
    public void addLabel(int dataPointIndex, int classIndex) {
        this.multiLabels[dataPointIndex].addLabel(classIndex);
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("numClasses=").append(numClasses).append("\n");
        sb.append(super.toString());
        sb.append("labels").append("\n");
        for (int i=0;i<numDataPoints;i++){
            sb.append(i).append(":").append(multiLabels[i]).append(",");
        }
        return sb.toString();
    }

    @Override
    public String getMetaInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMetaInfo());
        sb.append("type = ").append("sparse multi-label classification").append("\n");
        sb.append("number of classes = ").append(this.numClasses);
        return sb.toString();
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    @Override
    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
