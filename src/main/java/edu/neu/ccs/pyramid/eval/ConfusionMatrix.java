package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.util.Map;

/**
 * Created by chengli on 8/14/14.
 */
public class ConfusionMatrix {
    private int[][] matrix;
    private int numClasses;
    private Map<Integer, String> labelMap;

    public ConfusionMatrix(int numClasses, int[] labels, int[] predictions) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        int numDataPoints = labels.length;
        for (int i=0;i<numDataPoints;i++){
            int label = labels[i];
            int prediction = predictions[i];
            this.matrix[label][prediction] += 1;
        }
    }

    public ConfusionMatrix(int numClasses, Classifier classifier, ClfDataSet dataSet) {
        this(numClasses,dataSet.getLabels(),classifier.predict(dataSet));
    }

    public void setLabelMap(Map<Integer, String> labelMap) {
        this.labelMap = labelMap;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("\t");
        for (int k=0;k<this.numClasses;k++){
            sb.append(k).append("\t");
        }
        sb.append("\n");
        for (int i=0;i<this.numClasses;i++){
            sb.append(i).append("\t");
            for (int j=0;j<this.numClasses;j++){
                sb.append(matrix[i][j]).append("\t");
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * use real label
     * @return
     */
    public String prettyPrint() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.labelMap.toString()).append("\n");

        sb.append("\n");
        for (int i=0;i<this.numClasses;i++){
            sb.append(i).append("=>").append("\t");
            for (int j=0;j<this.numClasses;j++){
                sb.append(j).append(":").append(matrix[i][j]).append("\t");
            }
            sb.append("\n");
        }

        return sb.toString();

    }
}
