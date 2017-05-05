package edu.neu.ccs.pyramid.optimization.customize;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by yuyuxu on 5/1/17.
 * Model Iteration 2.
 * Refer write up formula (5).
 */
public class SupervisedEmbeddingTSNELoss {
    private DataSet X0;         // n x d, initial embedding matrix
    private DataSet Y0;         // n x 2, initial t-sne projection matrix
    private DataSet U;          // n x 2, user provided 2d plots for the data points
    private DataSet X;          // n x d, updated embedding matrix
    private DataSet Y;          // n x 2, updated t-sne projection matrix

    private double[] precision; // sigma per data point
    private double alpha;       // weight for first term
    private double beta;        // weight for second term
    private double gamma;       // weight for third term
    private double omega;       // weight for fourth term

    double pdenomi[];           // denominator for p
    double p[][];               // p_j|i
    double qdenomi[];           // denominator for q
    double q[][];               // q_ij


    private boolean flagValueCached;
    private boolean flagGradientCached;
    private boolean flagKLCached;
    private double cachedValue;
    private Vector cachedGradient;

    public SupervisedEmbeddingTSNELoss(DataSet X0, DataSet Y0, DataSet U,
                                       double[] precision, double alpha, double beta, double gamma, double omega) throws Exception {
        this.X0 = X0;
        this.Y0 = Y0;
        this.U = U;
        this.X = (DataSet)Serialization.deepCopy(X0);
        this.Y = (DataSet)Serialization.deepCopy(Y0);

        this.precision = precision.clone();
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.omega = omega;

        this.flagKLCached = false;
        int numData = X0.getNumDataPoints();
        pdenomi = new double[numData];
        p = new double[numData][numData];
        qdenomi = new double[numData];
        q = new double[numData][numData];

        this.flagValueCached = false;
        this.flagGradientCached = false;
        this.cachedGradient = new DenseVector(
                this.X.getNumDataPoints() * this.X.getNumFeatures() +
                this.Y.getNumDataPoints() * this.Y.getNumFeatures());
    }

    public Vector getParameters() {
        int numDataX = this.X.getNumDataPoints();
        int numFeaturesX = this.X.getNumFeatures();
        int numDataY = this.Y.getNumDataPoints();
        int numFeaturesY = this.Y.getNumFeatures();
        int offset = numDataX * numFeaturesX;

        Vector pVec = new DenseVector(numDataX * numFeaturesX + numDataY * numFeaturesY);
        for (int i = 0; i < numDataX; i++) {
            for (int j = 0; j < numFeaturesX; j++) {
                pVec.set(i * numFeaturesX + j, this.X.getRow(i).get(j));
            }
        }
        for (int i = 0; i < numDataY; i++) {
            for (int j = 0; j < numFeaturesY; j++) {
                pVec.set(i * numFeaturesY + j + offset, this.Y.getRow(i).get(j));
            }
        }

        return pVec;
    }

    public void setParameters(Vector parameters) {
        int numDataX = this.X.getNumDataPoints();
        int numFeaturesX = this.X.getNumFeatures();
        int numDataY = this.Y.getNumDataPoints();
        int numFeaturesY = this.Y.getNumFeatures();
        int offset = numDataX * numFeaturesX;

        for (int i = 0; i < numDataX; i++) {
            for (int j = 0; j < numFeaturesX; j++) {
                this.X.setFeatureValue(i, j, parameters.get(i * numFeaturesX + j));
            }
        }
        for (int i = 0; i < numDataY; i++) {
            for (int j = 0; j < numFeaturesY; j++) {
                this.Y.setFeatureValue(i, j, parameters.get(i * numFeaturesY + j + offset));
            }
        }
    
        this.flagValueCached = false;
        this.flagGradientCached = false;
        this.flagKLCached = false;
    }

    public double getValue() {
        if (this.flagValueCached) {
            return this.cachedValue;
        }

        this.flagValueCached = true;
        return this.cachedValue;
    }

    public Vector getGradient() {
        if (this.flagGradientCached) {
            return this.cachedGradient;
        }

        this.flagGradientCached = true;
        return this.cachedGradient;
    }

    private void updateKLVariables() {
        int numData = this.X.getNumDataPoints();
        for (int i = 0; i < numData; ++i) {
            pdenomi[i] = 0.0;
            for (int j = 0; j < numData; ++j) {
                if (j == i) {
                    continue;
                }
                Vector diff = this.X.getRow(j).minus(this.X.getRow(i));
                pdenomi[i] += Math.exp(-diff.dot(diff) * 0.5 * this.precision[i]);
            }
        }
    }
}
