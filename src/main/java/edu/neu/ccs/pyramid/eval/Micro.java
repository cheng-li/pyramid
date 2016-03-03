package edu.neu.ccs.pyramid.eval;

/**
 * Created by chengli on 3/2/16.
 */
public class Micro {
    private double tp;
    private double tn;
    private double fp;
    private double fn;
    private double f1;
    private double overlap;
    private double precision;
    private double recall;


    public Micro(MLConfusionMatrix confusionMatrix) {
        int numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
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
}
