package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.List;

/**
 * follow definition in
 * http://en.wikipedia.org/wiki/Receiver_operating_characteristic
 * Created by chengli on 10/3/14.
 */
public class PerClassMeasures {
    private int classIndex;
    private String className="unknown";
    private int positive;
    private int negative;
    private int truePositive;
    private int trueNegative;
    private int falsePositive;
    private int falseNegative;
    private double truePositiveRate;
    private double trueNegativeRate;
    private double falsePositiveRate;
    private double falseNegativeRate;
    private double accuracy;
    private double precision;
    private double recall;
    private double f1;


    /**
     * for single label dataset
     * @param confusionMatrix
     * @param classIndex
     */
    public PerClassMeasures(ConfusionMatrix confusionMatrix, int classIndex) {
        this.classIndex = classIndex;
        int numClasses = confusionMatrix.getNumClasses();
        int[][] matrix = confusionMatrix.getMatrix();
        for (int label=0;label<numClasses;label++){
            for (int prediction=0;prediction<numClasses;prediction++){
                int count = matrix[label][prediction];
                if (label==classIndex && prediction==classIndex){
                    truePositive += count;
                }
                if (label==classIndex && prediction!=classIndex){
                    falseNegative += count;
                }
                if (label!=classIndex && prediction==classIndex){
                    falsePositive += count;
                }
                if (label!=classIndex && prediction!=classIndex){
                    trueNegative += count;
                }
            }
        }

        positive = truePositive + falseNegative;
        negative = trueNegative + falsePositive;
        truePositiveRate = ((double)truePositive)/positive;
        trueNegativeRate = ((double)trueNegative)/negative;
        falsePositiveRate = ((double)falsePositive)/negative;
        falseNegativeRate = ((double)falseNegative)/positive;
        accuracy = ((double)(truePositive+trueNegative))/(positive+negative);
        precision = ((double)truePositive)/(truePositive+falsePositive);
        recall = truePositiveRate;
        f1 = FMeasure.f1(precision,recall);
    }

    /**
     * for single label dataset
     * @param confusionMatrix
     * @param classIndex
     * @param className
     */
    public PerClassMeasures(ConfusionMatrix confusionMatrix, int classIndex, String className){
        this(confusionMatrix, classIndex);
        this.className = className;
    }



    /**
     * for multi label dataset
     * @param multiLabels
     * @param predictions
     */
    public PerClassMeasures(MultiLabel[] multiLabels,
                            List<MultiLabel> predictions,
                            int classIndex){
        this(getBinaryConfusionMatrix(multiLabels,predictions,classIndex),1);
    }

    /**
     * for multi label dataset
     * @param multiLabels
     * @param predictions
     * @param classIndex
     * @param className
     */
    public PerClassMeasures(MultiLabel[] multiLabels,
                            List<MultiLabel> predictions,
                            int classIndex,
                            String className){
        this(multiLabels, predictions, classIndex);
        this.className = className;
    }

    public int getPositive() {
        return positive;
    }

    public int getNegative() {
        return negative;
    }

    public int getTruePositive() {
        return truePositive;
    }

    public int getTrueNegative() {
        return trueNegative;
    }

    public int getFalsePositive() {
        return falsePositive;
    }

    public int getFalseNegative() {
        return falseNegative;
    }

    public double getTruePositiveRate() {
        return truePositiveRate;
    }

    public double getTrueNegativeRate() {
        return trueNegativeRate;
    }

    public double getFalsePositiveRate() {
        return falsePositiveRate;
    }

    public double getFalseNegativeRate() {
        return falseNegativeRate;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1() {
        return f1;
    }

    @Override
    public String toString() {
        return "{" +
                "classIndex=" + classIndex +
                ", className=" + className +
                ", positive=" + positive +
                ", negative=" + negative +
                ", truePositive=" + truePositive +
                ", trueNegative=" + trueNegative +
                ", falsePositive=" + falsePositive +
                ", falseNegative=" + falseNegative +
                ", truePositiveRate=" + truePositiveRate +
                ", trueNegativeRate=" + trueNegativeRate +
                ", falsePositiveRate=" + falsePositiveRate +
                ", falseNegativeRate=" + falseNegativeRate +
                ", accuracy=" + accuracy +
                ", precision=" + precision +
                ", recall=" + recall +
                ", f1=" + f1 +
                '}';
    }

    private static ConfusionMatrix getBinaryConfusionMatrix(MultiLabel[] multiLabels, List<MultiLabel> predictions, int classIndex){
        int numDataPoints = multiLabels.length;
        int[] binaryLabels = new int[numDataPoints];
        int[] binaryPredictions = new int[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            if (multiLabels[i].matchClass(classIndex)){
                binaryLabels[i]=1;
            } else {
                binaryLabels[i]=0;
            }

            if (predictions.get(i).matchClass(classIndex)){
                binaryPredictions[i]=1;
            } else {
                binaryPredictions[i]=0;
            }
        }
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(2,binaryLabels,binaryPredictions);
        return confusionMatrix;
    }
}
