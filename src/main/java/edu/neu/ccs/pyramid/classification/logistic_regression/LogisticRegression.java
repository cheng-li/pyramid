package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 11/28/14.
 */
public class LogisticRegression implements ProbabilityEstimator {
    public Vector weights;

    @Override
    public int getNumClasses() {
        return 2;
    }

    @Override
    public int predict(Vector vector) {
        Vector withConstant = addConstant(vector);
        double product = this.weights.dot(withConstant);
//        System.out.println("product = "+product);
        if (product>=0){
            return 1;
        } else {
            return 0;
        }
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        Vector withConstant = addConstant(vector);
        double score = this.weights.dot(withConstant);
        double pro1;
        double logNominator = score;
        double[] exps = {0,score};
        double logDenominator = MathUtil.logSumExp(exps);
        pro1 = Math.exp(logNominator - logDenominator);
        double[] probs = new double[2];
        probs[0] = 1-pro1;
        probs[1] = pro1;
        return probs;
    }

    private static Vector addConstant(Vector vector){
        Vector vector1;
        if (vector.isDense()){
            vector1 = new DenseVector(vector.size()+1);
        } else {
            vector1 = new RandomAccessSparseVector(vector.size()+1);
        }
        vector1.set(0,1);
        for (Vector.Element element: vector.nonZeroes()){
            int index = element.index();
            double value = element.get();
            vector1.set(index+1, value);
        }
        return vector1;
    }
}
