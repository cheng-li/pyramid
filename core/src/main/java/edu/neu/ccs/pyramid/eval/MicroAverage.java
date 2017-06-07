package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.mahout.math.Vector;

/**
 * Based on
 * Koyejo, Oluwasanmi O., et al. "Consistent Multilabel Classification."
 * Advances in Neural Information Processing Systems. 2015.
 * convention: 0=TN, 1=TP, 2=FN, 3=FP
 * Created by chengli on 3/2/16.
 */
public class MicroAverage {

    private double f1;
    private double overlap;
    private double precision;
    private double recall;
    private double hammingLoss;


    public MicroAverage(MLConfusionMatrix confusionMatrix) {
        int numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
        double tp = 0;
        double tn = 0;
        double fp = 0;
        double fn = 0;
        DataSet entries = confusionMatrix.getEntries();
        for (int i=0;i<numDataPoints;i++){
            for (Vector.Element element: entries.getRow(i).nonZeroes()){
                double v = element.get();
                if (v==1){
                    tp += 1;
                } else if (v==2){
                    fn += 1;
                } else if (v==3){
                    fp += 1;
                }
            }
        }

        tn = numDataPoints*numClasses-tp-fp-fn;

        precision = Precision.precision(tp,fp);
        recall = Recall.recall(tp,fn);
        f1 = FMeasure.f1(tp,fp,fn);
        overlap = Overlap.overlap(tp,fp,fn);
        hammingLoss = HammingLoss.hammingLoss(tp,tn,fp,fn);
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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append("micro Jaccard index = ").append(overlap).append("\n");
        sb.append("micro Hamming loss = ").append(hammingLoss).append("\n");
        sb.append("micro F1 = ").append(f1).append("\n");
        sb.append("micro precision = ").append(precision).append("\n");
        sb.append("micro recall = ").append(recall).append("\n");
        return sb.toString();
    }
}
