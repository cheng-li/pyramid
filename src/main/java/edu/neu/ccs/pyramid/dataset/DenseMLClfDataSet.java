package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;

/**
 * Created by chengli on 9/27/14.
 */
public class DenseMLClfDataSet extends DenseDataSet implements MultiLabelClfDataSet{
    private int numClasses;
    private MultiLabel[] multiLabels;


    public DenseMLClfDataSet(int numDataPoints, int numFeatures, int numClasses){
        super(numDataPoints, numFeatures);
        this.numClasses=numClasses;
        this.multiLabels=new MultiLabel[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.multiLabels[i]= new MultiLabel(numClasses);
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
}
