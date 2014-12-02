package edu.neu.ccs.pyramid.classification.svm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.util.FastMath;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;

/**
 * Created by Rainicy on 11/29/14.
 */
public class SMOAction implements Classifier {

    private  int numClasses;
    // These are for svm models
    private double[] alphas;    // lagrange multipliers
    private double b;   // bias
    private String kernel;  // kernel

    // These are for training only
    private ClfDataSet dataSet;
    private double C;   // for C_SVC
    private double toler;   // KKT tolerance
    private int maxIter;    // maximum loops
    private int numDataPoints;
    private Map<Integer, Map<Integer, Double>> cacheKernel;   // kenel_(i,j)
    private Map<Integer, Double> errorCache;


    public SMOAction(double C, double toler, int maxIter, String kernel) {
        this.C = C;
        this.toler = toler;
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
    }


    public void train(ClfDataSet dataSet) {
        init(dataSet);

        int iter = 0;
        int numChanged = 0;
        boolean examineAll = true;
        while ((iter++ < maxIter) && ((numChanged > 0) || (examineAll))) {
            numChanged = 0;
            if (examineAll) {
                int[] entireIndexes = MathUtil.randomRange(0,numDataPoints);
                for (int i=0; i<entireIndexes.length; i++) {
                    numChanged += innerL(entireIndexes[i]);
                }
                System.out.println("Iter: " + iter + " on Entire Set | " +
                        " Alphas Pairs Changed: " + numChanged);
            }
            else {
                List<Integer> nonBoundIndexes = findNonBoundary();
                Collections.shuffle(nonBoundIndexes);
                for (int i=0; i<nonBoundIndexes.size(); i++) {
                    numChanged += innerL(nonBoundIndexes.get(i));
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

    private int innerL(int i) {
        double Ei = calEk(i);
        double yi = (double) dataSet.getLabels()[i];
        double alphai = alphas[i];
        if (((yi*Ei < -toler) && (alphai<C)) ||
                ((yi*Ei>toler) && (alphai>0))) {
            int j = selectJ(i,Ei);
            double Ej = calEk(j);

            double alphaIold = alphai;
            double alphaJold = alphas[j];
            double yj = (double) dataSet.getLabels()[j];

            double L, H;
            if (yi != yj) {
                L = FastMath.max(0, alphaJold-alphaIold);
                H = FastMath.min(C, C+alphaJold-alphaIold);
            }
            else {
                L = FastMath.max(0, alphaJold+alphaIold-C);
                H = FastMath.min(C, alphaJold+alphaIold);
            }
            if (L==H) return 0;

            double Kii = calKernel(i,i);
            double Kij = calKernel(i,j);
            double Kjj = calKernel(j,j);
            double eta = 2.0 * Kij - Kii - Kjj;
            if (eta >= 0) return 0;

            double newAlphaJ = alphaJold - yj*(Ei-Ej)/eta;
            if (newAlphaJ > H) newAlphaJ = H;
            if (newAlphaJ < L) newAlphaJ = L;
            if (FastMath.abs(alphaJold-newAlphaJ) < 0.001) return 0;

            double newAlphaI = alphaIold + yj*yi*(alphaJold-newAlphaJ);

            double bOld = b;
            double b1 = b - Ei - yi*(newAlphaI-alphaIold)*Kii - yj*(newAlphaJ-alphaJold)*Kij;
            double b2 = b - Ej - yi*(newAlphaI-alphaIold)*Kij - yj*(newAlphaJ-alphaJold)*Kjj;
            if ((0<newAlphaI) && (C>newAlphaI)) b = b1;
            else if ((0<newAlphaJ) && (C>newAlphaJ)) b = b2;
            else b = (b1 + b2) / 2.0;

            alphas[i] = newAlphaI;
            alphas[j] = newAlphaJ;

//            System.out.println(b);
            // update error cache
            updateErrorCache(i, alphaIold, newAlphaI, j, alphaJold, newAlphaJ, bOld);
            return 1;
        }
        return 0;
    }

    private void updateErrorCache(int i1, double alpha1, double a1, int i2, double alpha2, double a2, double bOld) {
        // according E = f(x) - y;
        double y1 = dataSet.getLabels()[i1];
        double y2 = dataSet.getLabels()[i2];
        double delta1 = (a1 - alpha1) * y1;
        double delta2 = (a2 - alpha2) * y2;

        for (int key : errorCache.keySet()) {
            double Ki1k = calKernel(i1, key);
            double Ki2k = calKernel(i2, key);
            double error = errorCache.get(key);
            error += delta1*Ki1k + delta2*Ki2k - bOld + b;
            errorCache.put(key, error);
        }
    }

    private int selectJ(int i, double Ei) {
        int maxK = -1;
        double maxDeltaE = -1;

        if (errorCache.size() > 1) {
            for (int k : errorCache.keySet()) {
                if (k==i) continue;
                double Ek = calEk(k);
                double deltaE = FastMath.abs(Ei - Ek);
                if(deltaE > maxDeltaE) {
                    maxDeltaE = deltaE;
                    maxK = k;
                }
            }
//            System.out.println(maxK);
            return maxK;
        }
        else {
            maxK = i;
            while (maxK == i) {
                maxK = (int)(FastMath.random()*((double)numDataPoints-1.0));
//                System.out.println(maxK);
            }
        }
//        System.out.println(maxK);
        return maxK;
    }

    /**
     * Calculate the error for given datapoint index.
     * @param k
     * @return
     */
    private double calEk(int k) {
        if (!errorCache.containsKey(k)) {
//            System.out.println(k);
            double yk = (double) dataSet.getLabels()[k];
            errorCache.put(k, functionX(k)-yk);
        }
        return errorCache.get(k);
    }

    private double functionX(int k) {
        double result = 0;
        for (int i=0; i<numDataPoints; i++) {
            double y = (double) dataSet.getLabels()[i];
            result += alphas[i] * y * calKernel(i,k);
        }
        result += b;
        return result;
    }

    private List<Integer> findNonBoundary() {
        List<Integer> nonBoundary = new LinkedList<Integer>();
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
            cacheKernel.put(i, new HashMap<Integer, Double>());
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
            double sigma = 0.1;
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

        for (int i=0; i<numDataPoints; i++) {
            double yi = (double) dataSet.getLabels()[i];
            double alpha = alphas[i];
            double k = getKernelValue(dataSet.getRow(i), vector, kernel);
            hypothese += yi * alpha * k;
        }
        hypothese += b;

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
        str = "C= " + C + ",  toler= " + toler;
        return str;
    }
}


