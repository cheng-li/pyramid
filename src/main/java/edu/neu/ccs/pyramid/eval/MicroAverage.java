package edu.neu.ccs.pyramid.eval;

/**
 * Based on
 * Koyejo, Oluwasanmi O., et al. "Consistent Multilabel Classification."
 * Advances in Neural Information Processing Systems. 2015.
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
        MLConfusionMatrix.Entry[][] entries = confusionMatrix.getEntries();
        for (int i=0;i<numDataPoints;i++){
            for (int l=0;l<numClasses;l++){
                MLConfusionMatrix.Entry entry = entries[i][l];
                switch (entry){
                    case TP:
                        tp += 1;
                        break;
                    case FP:
                        fp += 1;
                        break;
                    case TN:
                        tn += 1;
                        break;
                    case FN:
                        fn += 1;
                        break;
                }
            }
        }

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
        sb.append("micro overlap=").append(overlap).append(",").append("\n");
        sb.append("micro Hamming loss=").append(hammingLoss).append(",").append("\n");
        sb.append("micro F1=").append(f1).append(",").append("\n");
        sb.append("micro precision=").append(precision).append(",").append("\n");
        sb.append("micro recall=").append(recall).append("\n");
        return sb.toString();
    }
}
