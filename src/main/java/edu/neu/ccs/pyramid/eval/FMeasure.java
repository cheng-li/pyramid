package edu.neu.ccs.pyramid.eval;

/**
 * follow the definition in http://en.wikipedia.org/wiki/Precision_and_recall
 * Created by chengli on 10/2/14.
 */
public class FMeasure {

    /**
     *
     * @param precision
     * @param recall
     * @return
     */
    public static double f1(double precision, double recall){
        return fBeta(precision,recall,1);
    }

    /**
     *
     * @param precision
     * @param recall
     * @param beta
     * @return
     */
    public static double fBeta(double precision, double recall, double beta){
        return (1+beta*beta)*precision*recall/(beta*beta*precision + recall);
    }
}
