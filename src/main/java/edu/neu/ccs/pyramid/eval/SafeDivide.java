package edu.neu.ccs.pyramid.eval;

/**
 * Created by chengli on 3/2/16.
 */
public class SafeDivide {

    /**
     * allow 0/0 to return a default value
     * @param numerator a
     * @param denominator b
     * @param defaultValue 0/0
     * @return a/b
     */
    public static double divide(double numerator, double denominator, double defaultValue){
        if (denominator==0){
            if (numerator==0){
                return defaultValue;
            } else {
                throw new ArithmeticException("non-zero number divided by 0");
            }
        }
        return numerator/denominator;
    }
}
