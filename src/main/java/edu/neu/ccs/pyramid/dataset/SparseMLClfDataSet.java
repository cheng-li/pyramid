package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 9/27/14.
 */
public class SparseMLClfDataSet extends SparseDataSet implements MultiLabelClfDataSet{
    private int numClasses;
    private MultiLabel[] multiLabels;

    public SparseMLClfDataSet(int numDataPoints, int numFeatures, int numClasses){
        super(numDataPoints, numFeatures);
        this.numClasses=numClasses;
        this.multiLabels=new MultiLabel[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.multiLabels[i]= new MultiLabel();
        }
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
}
