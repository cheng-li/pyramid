package edu.neu.ccs.pyramid.eval;

import org.apache.commons.math3.util.FastMath;

/**
 * Created by chengli on 10/8/15.
 */
public class Entropy {
    /**
     * by default, use e based log
     * @param distribution
     * @return
     */
    public static double entropy(double[] distribution){
        double res = 0;
        for (double x: distribution){
            if (x!=0){
                res += -x* Math.log(x);
            }

        }
        return res;
    }

    public static double entropy2Based(double[] distribution){
        double res = 0;
        for (double x: distribution){
            if (x!=0){
                res += -x* FastMath.log(2, x);
            }

        }
        return res;
    }
}
