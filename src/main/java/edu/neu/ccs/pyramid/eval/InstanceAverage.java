package edu.neu.ccs.pyramid.eval;

import java.util.stream.IntStream;

/**
 * Based on
 * Koyejo, Oluwasanmi O., et al. "Consistent Multilabel Classification."
 * Advances in Neural Information Processing Systems. 2015.
 * Created by chengli on 3/3/16.
 */
public class InstanceAverage {
    private double f1;
    private double overlap;
    private double precision;
    private double recall;
    private double hammingLoss;
    private double accuracy;

    public InstanceAverage(MLConfusionMatrix confusionMatrix) {
        int numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
        MLConfusionMatrix.Entry[][] entries = confusionMatrix.getEntries();
        double[] tpArray = new double[numDataPoints];
        double[] tnArray = new double[numDataPoints];
        double[] fpArray = new double[numDataPoints];
        double[] fnArray = new double[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            for (int l=0;l<numClasses;l++){
                MLConfusionMatrix.Entry entry = entries[i][l];
                switch (entry){
                    case TP:
                        tpArray[i] += 1;
                        break;
                    case FP:
                        fpArray[i] += 1;
                        break;
                    case TN:
                        tnArray[i] += 1;
                        break;
                    case FN:
                        fnArray[i] += 1;
                        break;
                }
            }
            tpArray[i] /= numClasses;
            tnArray[i] /= numClasses;
            fpArray[i] /= numClasses;
            fnArray[i] /= numClasses;
        }

        precision = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i->Precision.precision(tpArray[i],fpArray[i]))
                .average().getAsDouble();

        recall = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i->Recall.recall(tpArray[i],fnArray[i]))
                .average().getAsDouble();

        f1 = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i->FMeasure.f1(tpArray[i],fpArray[i],fnArray[i]))
                .average().getAsDouble();

        overlap = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i->Overlap.overlap(tpArray[i],fpArray[i],fnArray[i]))
                .average().getAsDouble();

        hammingLoss = IntStream.range(0,numDataPoints).parallel()
                .mapToDouble(i->HammingLoss.hammingLoss(tpArray[i],tnArray[i],
                        fpArray[i],fnArray[i]))
                .average().getAsDouble();

        accuracy = IntStream.range(0,numDataPoints).parallel()
                .filter(i->correct(entries[i])).count()/(double)numDataPoints;

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

    /**
     * Judge whether a prediction is completely correct based on a row of confusion matrix
     * Using raw entries is more stable than using double numbers.
     * @param dataEntry a row
     * @return
     */
    private boolean correct(MLConfusionMatrix.Entry[] dataEntry){
        for (int l=0;l<dataEntry.length;l++){
            MLConfusionMatrix.Entry entry = dataEntry[l];
            if (entry== MLConfusionMatrix.Entry.FP || entry== MLConfusionMatrix.Entry.FN){
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("InstanceAverage{").append("\n");
        sb.append("accuracy=").append(accuracy).append(",").append("\n");
        sb.append("overlap=").append(overlap).append(",").append("\n");
        sb.append("hammingLoss=").append(hammingLoss).append(",").append("\n");
        sb.append("f1=").append(f1).append(",").append("\n");
        sb.append("precision=").append(precision).append(",").append("\n");
        sb.append("recall=").append(recall).append("\n");
        sb.append('}');
        return sb.toString();
    }
}
