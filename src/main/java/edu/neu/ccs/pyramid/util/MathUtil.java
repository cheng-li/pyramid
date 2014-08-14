package edu.neu.ccs.pyramid.util;

/**
 * Created by chengli on 8/14/14.
 */
public class MathUtil {
    /**
     * calculate log(exp(x1)+exp(x2)+...)
     * @return
     */
    public static double logSumExp(double[] arr){
        double maxElement = arr[0];
        for (double number: arr){
            if (number > maxElement){
                maxElement = number;
            }
        }
        double sum = 0;
        for (double number: arr){
            sum += Math.exp(number - maxElement);
        }
        return Math.log(sum) + maxElement;
    }

    /**
     *
     * @param arr
     * @return L1 norm of an array
     */
    public static double l1Norm(double[] arr){
        double norm = 0;
        for (double number: arr){
            norm += Math.abs(number);
        }
        return norm;
    }

    public static double l2Norm(double[] arr){
        double norm = 0;
        for (double number: arr){
            norm += Math.pow(number, 2);
        }
        return Math.sqrt(norm);
    }

    public static double maxNorm(double[] arr){
        double norm = 0;
        for (double number: arr){
            double abs = Math.abs(number);
            if (abs>norm){
                norm = abs;
            }
        }
        return norm;
    }

    /**
     *
     * @param distribution  assume distribution sums up to 1
     * @return
     */
    public static double entropy(double[] distribution){
        double entropy = 0;
        for (double prob: distribution){
            if (prob!=0){
                entropy -= prob*Math.log(prob)/Math.log(2);
            }
        }
        return entropy;
    }

    public static double arraySum(double[] arr){
        double sum = 0;
        for(double num:arr){
            sum += num;
        }
        return sum;
    }

    public static float arraySum(float[] arr){
        float sum = 0;
        for(float num:arr){
            sum += num;
        }
        return sum;
    }
}
