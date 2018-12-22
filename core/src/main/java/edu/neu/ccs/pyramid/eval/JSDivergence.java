package edu.neu.ccs.pyramid.eval;

public class JSDivergence {

    /**
     *
     * @param distributions each row is a distribution
     * @return
     */
    public static double js(double[][] distributions){
        double[] average = new double[distributions[0].length];
        for (int i=0;i<distributions.length;i++){
            for (int j=0;j<distributions[0].length;j++){
                average[j] += distributions[i][j];
            }
        }

        for (int j=0;j<distributions[0].length;j++){
            average[j] /= distributions.length;
        }

        double sum = 0;
        for (int i=0;i<distributions.length;i++){
            sum += KLDivergence.kl(distributions[i],average);
        }

        return sum/distributions.length;
    }

    /**
     * divided by log(n)
     * @param distributions
     * @return
     */
    public static double jsScaled(double[][] distributions){
        return js(distributions)/Math.log(distributions.length);
    }

}
