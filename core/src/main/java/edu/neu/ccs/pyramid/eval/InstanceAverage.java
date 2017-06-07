package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Based on
 * Koyejo, Oluwasanmi O., et al. "Consistent Multilabel Classification."
 * Advances in Neural Information Processing Systems. 2015.
 * convention: 0=TN, 1=TP, 2=FN, 3=FP
 * Created by chengli on 3/3/16.
 */
public class InstanceAverage {
    private double f1;
    private double overlap;
    private double precision;
    private double recall;
    private double hammingLoss;
    private double accuracy;

    public InstanceAverage(int numClasses, MultiLabel trueLabel, MultiLabel prediction){
        this(new MLConfusionMatrix(numClasses,toArray(trueLabel),toArray(prediction)));
    }

    private static MultiLabel[] toArray(MultiLabel multiLabel){
        return new MultiLabel[]{multiLabel};
    }

    public InstanceAverage(MLConfusionMatrix confusionMatrix) {
        int numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
        DataSet entries = confusionMatrix.getEntries();
        double[] tpArray = new double[numDataPoints];
        double[] tnArray = new double[numDataPoints];
        double[] fpArray = new double[numDataPoints];
        double[] fnArray = new double[numDataPoints];
        IntStream.range(0, numDataPoints).parallel().forEach(i->{
            for (Vector.Element element: entries.getRow(i).nonZeroes()){
                double v = element.get();
                if (v==1){
                    tpArray[i] += 1;
                } else if (v==2){
                    fnArray[i] += 1;
                } else if (v==3){
                    fpArray[i] += 1;
                }
            }
            tnArray[i]  = numClasses - entries.getRow(i).getNumNonZeroElements();
            tpArray[i] /= numClasses;
            tnArray[i] /= numClasses;
            fpArray[i] /= numClasses;
            fnArray[i] /= numClasses;
        });

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
                .filter(i->correct(entries.getRow(i))).count()/(double)numDataPoints;

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

    public double getAccuracy() {
        return accuracy;
    }

    /**
     * Judge whether a prediction is completely correct based on a row of confusion matrix
     * Using raw entries is more stable than using double numbers.
     * @param dataEntry a row
     * @return
     */
    private boolean correct(Vector dataEntry){
        for (Vector.Element element: dataEntry.nonZeroes()){
            double v = element.get();
            if (v==2||v==3){
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append("instance subset accuracy = ").append(accuracy).append("\n");
        sb.append("instance Jaccard index = ").append(overlap).append("\n");
        sb.append("instance Hamming loss = ").append(hammingLoss).append("\n");
        sb.append("instance F1 = ").append(f1).append("\n");
        sb.append("instance precision = ").append(precision).append("\n");
        sb.append("instance recall = ").append(recall).append("\n");
        return sb.toString();
    }
}
