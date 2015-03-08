package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;

/**
 * Created by chengli on 8/14/14.
 */
public class ConfusionMatrix {
    /**
     * row: label, column:prediction
     */
    private int[][] matrix;
    private int numClasses;
    private LabelTranslator labelTranslator;
    private int[] total;

    public ConfusionMatrix(int numClasses, int[] labels, int[] predictions) {
        this.numClasses = numClasses;
        this.matrix = new int[numClasses][numClasses];
        this.total = new int[numClasses];
        int numDataPoints = labels.length;
        for (int i=0;i<numDataPoints;i++){
            int label = labels[i];
            int prediction = predictions[i];
            this.matrix[label][prediction] += 1;
            this.total[label] += 1;
        }
    }

    public ConfusionMatrix(Classifier classifier, ClfDataSet dataSet) {
        this(dataSet.getNumClasses(),dataSet.getLabels(),classifier.predict(dataSet));
        this.labelTranslator = dataSet.getLabelTranslator();
    }

    public int[][] getMatrix() {
        return matrix;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public String printWithIntLabels() {
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
    public String printWithExtLabels() {
        StringBuilder sb = new StringBuilder();

        sb.append("\n");
        for (int i=0;i<this.numClasses;i++){
            sb.append(this.labelTranslator.toExtLabel(i)).append("(").append(this.total[i]).append(")")
                    .append("=>").append("   ");
            for (int j=0;j<this.numClasses;j++){
                sb.append(this.labelTranslator.toExtLabel(j)).append(":").append(matrix[i][j]).append("   ");
            }
            sb.append("\n");
        }

        return sb.toString();

    }
}
