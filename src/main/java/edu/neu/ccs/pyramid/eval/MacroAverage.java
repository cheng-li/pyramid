package edu.neu.ccs.pyramid.eval;

import java.util.Arrays;

/**
 * Created by chengli on 3/24/16.
 */
public class MacroAverage {
    private double f1;
    private double overlap;
    private double precision;
    private double recall;
    private double hammingLoss;
    // per class counts
    private int[] labelWiseTP;
    private int[] labelWiseTN;
    private int[] labelWiseFP;
    private int[] labelWiseFN;
    // per class measures
    private double[] labelWisePrecision;
    private double[] labelWiseRecall;
    private double[] labelWiseOverlap;
    private double[] labelWiseF1;
    private double[] labelWiseHammingLoss;


    public MacroAverage(MLConfusionMatrix confusionMatrix) {
        int numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
        MLConfusionMatrix.Entry[][] entries = confusionMatrix.getEntries();
        this.labelWiseTP = new int[numClasses];
        this.labelWiseTN = new int[numClasses];
        this.labelWiseFP = new int[numClasses];
        this.labelWiseFN = new int[numClasses];

        this.labelWisePrecision = new double[numClasses];
        this.labelWiseRecall = new double[numClasses];
        this.labelWiseOverlap = new double[numClasses];
        this.labelWiseF1 = new double[numClasses];
        this.labelWiseHammingLoss = new double[numClasses];


        for (int l=0;l<numClasses;l++){
            for (int i=0;i<numDataPoints;i++){
                MLConfusionMatrix.Entry entry = entries[i][l];
                switch (entry){
                    case TP:
                        labelWiseTP[l] += 1;
                        break;
                    case FP:
                        labelWiseFP[l] += 1;
                        break;
                    case TN:
                        labelWiseTN[l] += 1;
                        break;
                    case FN:
                        labelWiseFN[l] += 1;
                        break;
                }
            }
            double tp = ((double) labelWiseTP[l])/numDataPoints;
            double tn = ((double) labelWiseTN[l])/numDataPoints;
            double fp = ((double) labelWiseFP[l])/numDataPoints;
            double fn = ((double) labelWiseFN[l])/numDataPoints;

            labelWisePrecision[l] = Precision.precision(tp,fp);
            labelWiseRecall[l] = Recall.recall(tp,fn);
            labelWiseF1[l] = FMeasure.f1(tp,fp,fn);
            labelWiseOverlap[l] = Overlap.overlap(tp,fp,fn);
            labelWiseHammingLoss[l] = HammingLoss.hammingLoss(tp,tn,
                    fp,fn);
        }

        precision = Arrays.stream(labelWisePrecision).average().getAsDouble();

        recall = Arrays.stream(labelWiseRecall).average().getAsDouble();

        f1 = Arrays.stream(labelWiseF1).average().getAsDouble();

        overlap = Arrays.stream(labelWiseOverlap).average().getAsDouble();

        hammingLoss = Arrays.stream(labelWiseHammingLoss).average().getAsDouble();
    }

    public double getF1() {
        return f1;
    }

    public double getOverlap() {
        return overlap;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getHammingLoss() {
        return hammingLoss;
    }

    public int[] getLabelWiseTP() {
        return labelWiseTP;
    }

    public int[] getLabelWiseTN() {
        return labelWiseTN;
    }

    public int[] getLabelWiseFP() {
        return labelWiseFP;
    }

    public int[] getLabelWiseFN() {
        return labelWiseFN;
    }

    public double[] getLabelWisePrecision() {
        return labelWisePrecision;
    }

    public double[] getLabelWiseRecall() {
        return labelWiseRecall;
    }

    public double[] getLabelWiseOverlap() {
        return labelWiseOverlap;
    }

    public double[] getLabelWiseF1() {
        return labelWiseF1;
    }

    public double[] getLabelWiseHammingLoss() {
        return labelWiseHammingLoss;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append("macro overlap=").append(overlap).append(",").append("\n");
        sb.append("macro Hamming loss=").append(hammingLoss).append(",").append("\n");
        sb.append("macro F1=").append(f1).append(",").append("\n");
        sb.append("macro precision=").append(precision).append(",").append("\n");
        sb.append("macro recall=").append(recall).append("\n");
        return sb.toString();
    }
}
