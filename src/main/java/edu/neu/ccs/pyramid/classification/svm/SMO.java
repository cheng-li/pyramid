package edu.neu.ccs.pyramid.classification.svm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.util.FastMath;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;

/**
 * Created by Rainicy on 11/28/14.
 */
public class SMO implements Classifier {

    private  int numClasses;
    // These are for svm models
    private double[] alphas;    // lagrange multipliers
    private double b;   // bias
    private String kernel;  // kernel
    private Map<Integer, Double> nonBoundaries;  //

    // These are for training only
    private ClfDataSet dataSet;
    private double C;   // for C_SVC
    private double toler;   // KKT tolerance
    private double eps;
    private int maxIter;    // maximum loops
    private int numDataPoints;
    private Map<Integer, Map<Integer, Double>> cacheKernel;   // kenel_(i,j)
    private Map<Integer, Double> errorCache;

    // test
    private int class1 = 0;
    private int class2 = 0;
    private int class3 = 0;


    public SMO(double C, double toler, double eps, int maxIter, String kernel) {
        this.C = C;
        this.toler = toler;
        this.eps = eps;
        this.maxIter = maxIter;
        this.kernel = kernel;
    }


    private void init(ClfDataSet dataSet) {
        this.numClasses = dataSet.getNumClasses();
        this.numDataPoints = dataSet.getNumDataPoints();
        this.dataSet = dataSet;
        this.alphas = MathUtil.zeros(numDataPoints);
        this.b = 0.0;
        this.cacheKernel = new HashMap<>();
        this.errorCache = new HashMap<>();
        this.nonBoundaries = new HashMap<>();
    }


    public void train(ClfDataSet dataSet) {
        init(dataSet);

        int iter = 0;
        int numChanged = 0;
        boolean examineAll = true;
        while ((iter++ < maxIter) && (numChanged > 0 || examineAll)) {
            numChanged = 0;
            if (examineAll) {
                // shuffle the order, not necessary for other dataset.
                int[] entireIndexes = MathUtil.randomRange(0,numDataPoints);
                for (int i=0; i<entireIndexes.length; i++) {
                    numChanged += examineExample(entireIndexes[i]);
                }
                System.out.println("Iter: " + iter + " on Entire Set | " +
                        " Alphas Pairs Changed: " + numChanged);
            }
            else {
                List<Integer> nonBoundIndexes = findNonBoundary();
                Collections.shuffle(nonBoundIndexes);
                for (int i=0; i<nonBoundIndexes.size(); i++) {
                    numChanged += examineExample(nonBoundIndexes.get(i));
                }
                System.out.println("Iter: " + iter + " on Non-Boundary Set | " +
                        " Alphas Pairs Changed: " + numChanged);
            }

            if (examineAll) {
                examineAll = false;
            }
            else if (numChanged == 0) {
                examineAll = true;
            }
        }
    }

    private int examineExample(int i2) {
        int y2 = dataSet.getLabels()[i2];
        double alpha2 = alphas[i2];
        double E2 = calEk(i2);
        double r2 = E2 * y2;
        if (((r2 < -this.toler) && (alpha2 < this.C)) ||
                (r2 > this.toler) && (alpha2 > 0)) {
            List<Integer> nonBoundIndexes = findNonBoundary();
            if (nonBoundIndexes.size() > 1) {
                int i1 = selectJ(i2, E2, nonBoundIndexes);
                if (takeStep(i1, i2, y2, alpha2, E2)) {
                    class1++;
                    return 1;
                }
            }
            // random choose nonbound points
            Collections.shuffle(nonBoundIndexes);
            for (int k=0; k<nonBoundIndexes.size(); k++) {
                int i1 = nonBoundIndexes.get(k);
                if (takeStep(i1, i2, y2, alpha2, E2)) {
                    class2++;
                    return 1;
                }
            }
            int[] entireIndexes = MathUtil.randomRange(0,numDataPoints);
            for (int k=0; k<entireIndexes.length; k++) {
//                System.out.println(k + ": " + entireIndexes[k]);
                int i1 = entireIndexes[k];
                if (takeStep(i1, i2, y2, alpha2, E2)) {
                    class3++;
                    return 1;
                }
            }
        }
        return 0;
    }
    /**
     * Innter loop to update alpha_i and alpha_j;
     * @param i1, i2, y2, alpha2, E2
     * @return
     */
    private boolean takeStep(int i1, int i2, double y2, double alpha2, double E2) {

        if (i1 == i2) {
            return false;
        }

        double y1 = dataSet.getLabels()[i1];
        double alpha1 = alphas[i1];
        double E1 = calEk(i1);
        double s = y1*y2;

        double H, L;
        if (y1 != y2) {
            L = FastMath.max(0, alpha2 - alpha1);
            H = FastMath.min(C, alpha2 - alpha1 + C);
        }
        else {
            L = FastMath.max(0, alpha2 + alpha1 - C);
            H = FastMath.min(C, alpha2 + alpha1);
        }
        if (L == H) {
            return false;
        }

        // Step 3: eta
        double K11 = calKernel(i1,i1);
        double K12 = calKernel(i1,i2);
        double K22 = calKernel(i2,i2);
        double eta = K11 + K22 - 2*K12;

        double a1, a2;
        if (eta > 0) {
            // Step 4: update j
            a2 = alpha2 + y2 * (E1 - E2) / eta;
            // Step 5: clip alpha j
            if (a2 < L) {
                a2 = L;
            }
            else if (a2 > H) {
                a2 = H;
            }
        }
        else {
            double f1 = y1 * (E1 + b) - alpha1*K11 - s*alpha2*K12;
            double f2 = y2 * (E2 + b) - s*alpha1*K12 - alpha2*K22;
            double L1 = alpha1 + s*(alpha2-L);
            double H1 = alpha1 + s*(alpha2-H);
            double Lobj = L1*f1 + L*f2 + 0.5*L1*L1*K11 + 0.5*L*L*K22 + s*L*L1*K12;
            double Hobj = H1*f1 + H*f2 + 0.5*H1*H1*K11 + 0.5*H*H*K22 + s*H*H1*K12;
            if (Lobj < Hobj - eps) {
                a2 = L;
            }
            else if (Lobj > Hobj + eps) {
                a2 = H;
            }
            else {
                a2 = alpha2;
            }
        }

        if (FastMath.abs(alpha2 - a2) < eps*(alpha2+a2+eps)) {
            return false;
        }

        // Step 7: update alpha_i
        a1 = alpha1 + (s * (alpha2 - a2));

        // Step 8: update b
        double b1 = b + E1 + (a1-alpha1)*y1*K11 + (a2-alpha2)*y2*K12;
        double b2 = b + E2 + (a1-alpha1)*y1*K12 + (a2-alpha2)*y2*K22;
        double bOld = b;
        if (a1 > 0 && a1 < C) {
            b = b1;
        }
        else if (a2 > 0 && a2 < C) {
            b = b2;
        }
        else {
            b = (b1 + b2) / 2.0;
        }

        alphas[i1] = a1;
        alphas[i2] = a2;

        if (a1 != 0) nonBoundaries.put(i1, a1);
        if (a2 != 0) nonBoundaries.put(i2, a2);
        if (a1 == 0) nonBoundaries.remove(i1);
        if (a2 == 0) nonBoundaries.remove(i2);

        // update error cache
        updateErrorCache(i1, alpha1, a1, i2, alpha2, a2, bOld);
        return true;
    }

    private void updateErrorCache(int i1, double alpha1, double a1, int i2, double alpha2, double a2, double bOld) {
        // according E = f(x) - y;
        double y1 = dataSet.getLabels()[i1];
        double y2 = dataSet.getLabels()[i2];
        double delta1 = (a1 - alpha1) * y1;
        double delta2 = (a2 - alpha2) * y2;

        for (Map.Entry<Integer, Double> entry : errorCache.entrySet()) {
            int key = entry.getKey();
            double error = entry.getValue();
            double Ki1k = calKernel(i1, key);
            double Ki2k = calKernel(i2, key);
            error += delta1*Ki1k + delta2*Ki2k + bOld - b;
            errorCache.put(key, error);
        }
    }

    /**
     * Select alpha_j by max(error);
     * @param i
     * @param Ei
     * @param nonBoundIndexes
     * @return
     */
    private int selectJ(int i, double Ei, List<Integer> nonBoundIndexes) {
        double maxDelta = 0;
        int j = -1;

        for (int k=0; k<nonBoundIndexes.size(); k++) {
            if (i == nonBoundIndexes.get(k))
                continue;
            double Ek = calEk(nonBoundIndexes.get(k));
            double delta = FastMath.abs(Ei - Ek);
            if (delta > maxDelta) {
                j = nonBoundIndexes.get(k);
                maxDelta = delta;
            }
        }
        return j;
    }


    /**
     * Calculate the error for given datapoint index.
     * @param k
     * @return
     */
    private double calEk(int k) {
        if (!errorCache.containsKey(k)) {
            double yk = (double) dataSet.getLabels()[k];
            errorCache.put(k, functionX(k)-yk);
        }
        return errorCache.get(k);
    }

    private double functionX(int k) {
        double result = 0;
        for (Map.Entry<Integer, Double> entry : nonBoundaries.entrySet()) {
            int key = entry.getKey();
            double alpha = entry.getValue();
            double y = (double) dataSet.getLabels()[key];
            result += alpha * y * calKernel(key,k);
        }
//        for (int i=0; i<numDataPoints; i++) {
//            double y = (double) dataSet.getLabels()[i];
//            result += alphas[i] * y * calKernel(i,k);
//        }
        result -= b;
        return result;
    }

    private List<Integer> findNonBoundary() {
        List<Integer> nonBoundary = new ArrayList<>();
        for (int i=0; i<alphas.length; i++) {
            double alpha = alphas[i];
            if ((alpha>0) && (alpha<C)) {
                nonBoundary.add(i);
            }
        }
        return nonBoundary;
    }

    private double calKernel(int i, int j) {
        if (!cacheKernel.containsKey(i)) {
            Vector vectorI = dataSet.getRow(i);
            Vector vectorJ = dataSet.getRow(j);
            cacheKernel.put(i, new HashMap<>());
            cacheKernel.get(i).put(j, getKernelValue(vectorI, vectorJ, kernel));
        }
        else if (!cacheKernel.get(i).containsKey(j)) {
            Vector vectorI = dataSet.getRow(i);
            Vector vectorJ = dataSet.getRow(j);
            cacheKernel.get(i).put(j, getKernelValue(vectorI, vectorJ, kernel));
        }
        return cacheKernel.get(i).get(j);
    }

    private double getKernelValue(Vector vectorI, Vector vectorJ, String kernel) {
        double result;
        if (kernel == "linear") {
            result = vectorI.dot(vectorJ);
        }
        else if (kernel == "rbf") {
            // default sigma
            double sigma = 4;
            Vector diffVector = vectorI.minus(vectorJ);
            double diffValue = diffVector.dot(diffVector);
            diffValue = (-0.5) * diffValue / FastMath.pow(sigma, 2);
            result = FastMath.exp(diffValue);
        }
        else {
            throw new RuntimeException("The kernel cannot be recognized.");
        }
        return result;
    }

    @Override
    public int predict(Vector vector) {
        double hypothese = 0.0;
        for (Map.Entry<Integer, Double> entry : nonBoundaries.entrySet()) {
            int index = entry.getKey();
            double alpha = entry.getValue();
            double yi = (double) dataSet.getLabels()[index];
            double k = getKernelValue(dataSet.getRow(index), vector, kernel);
            hypothese += yi * alpha * k;
        }
        hypothese -= b;

        if (hypothese >= 0) {
            return 1;
        }
        return -1;
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }

    @Override
    public int[] predict(ClfDataSet dataSet) {
        int[] results = new int[dataSet.getNumDataPoints()];
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            results[i] = predict(dataSet.getRow(i));
        }
        return results;
    }

    @Override
    public void serialize(File file) throws Exception {

    }

    @Override
    public void serialize(String file) throws Exception {

    }
    public String toString() {
        String str = new String();
        str = "C= " + C + ",  toler= " + toler + ", eps= " + eps
                + ", kernel= " + kernel + ", #sv: " + nonBoundaries.size();
        str += "class1: " + class1 + ", class2: " + class2 + ", class3: " + class3;
        return str;
    }
}


