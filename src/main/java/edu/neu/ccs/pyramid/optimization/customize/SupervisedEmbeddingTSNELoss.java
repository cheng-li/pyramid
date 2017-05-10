package edu.neu.ccs.pyramid.optimization.customize;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by yuyuxu on 5/1/17.
 * Model Iteration 2.
 * Refer write up formula (5).
 */
public class SupervisedEmbeddingTSNELoss implements Optimizable.ByGradientValue {
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
    double p[][];               // p[i][j] = p_i|j
    double qdenomi;             // denominator for q
    double q[][];               // q_ij
    double p_scale[];           // scale term for p_j|i in gradient term

    private boolean flagValueCached;
    private boolean flagGradientCached;
    private boolean flagKLVariableCached;
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

        this.flagKLVariableCached = false;
        int numData = X0.getNumDataPoints();
        this.pdenomi = new double[numData];
        this.p = new double[numData][numData];
        this.q = new double[numData][numData];
        this.p_scale = new double[numData];

        this.flagValueCached = false;
        this.flagGradientCached = false;
        this.cachedGradient = new DenseVector(
                this.X.getNumDataPoints() * this.X.getNumFeatures() +
                this.Y.getNumDataPoints() * this.Y.getNumFeatures());
    }

    public DataSet getX() {
        return this.X;
    }

    public DataSet getY() {
        return this.Y;
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
        this.flagKLVariableCached = false;
    }

    public double getValue() {
        if (this.flagValueCached) {
            return this.cachedValue;
        }

        if (!this.flagKLVariableCached) {
            updateKLVariables();
        }

        int numData = this.X.getNumDataPoints();
        this.cachedValue = IntStream.range(0, numData)
                .parallel()
                .mapToDouble(this::updateValue).sum();
        this.flagValueCached = true;

        return this.cachedValue;
    }

    private double updateValue(int i) {
        int numData = this.X.getNumDataPoints();
        double res = 0.0;

        // first term
        Vector diff1 = this.X.getRow(i).minus(this.X0.getRow(i));
        res += diff1.dot(diff1) * this.alpha;

        // second term
        Vector diff2 = this.Y.getRow(i).minus(this.Y0.getRow(i));
        res += diff2.dot(diff2) * this.beta;

        double thirdTermLoss = 0.0;
        for (int j = 0; j < numData; ++j) {
            if (i == j) {
                continue;
            }
            Vector diff_proj = this.Y.getRow(i).minus(this.Y.getRow(j));
            Vector diff_user = this.U.getRow(i).minus(this.U.getRow(j));
            double diff = diff_proj.dot(diff_proj) - diff_user.dot(diff_user);
            thirdTermLoss += diff * diff;
        }
        res += thirdTermLoss * this.gamma;

        double klTermLoss = 0.0;
        for (int j = 0; j < numData; ++j) {
            if (i == j) {
                continue;
            }
            double p_ij = (this.p[i][j] + this.p[j][i]) * 0.5 / (double) numData;
            klTermLoss += p_ij * (Math.log(p_ij) - Math.log(this.q[i][j]));
        }
        res += klTermLoss * this.omega;

        return res;
    }

    public Vector getGradient() {
        if (this.flagGradientCached) {
            return this.cachedGradient;
        }

        if (!this.flagKLVariableCached) {
            updateKLVariables();
        }

        int numData = this.X.getNumDataPoints();
        int numFeaturesX = this.X.getNumFeatures();
        int numFeaturesY = this.Y.getNumFeatures();
        int offset = this.X.getNumDataPoints() * this.X.getNumFeatures();
        List<Vector> gradientX = IntStream.range(0, numData)
                .parallel()
                .mapToObj(this::updateGradientX).collect(Collectors.toList());
        List<Vector> gradientY = IntStream.range(0, numData)
                .parallel()
                .mapToObj(this::updateGradientY).collect(Collectors.toList());
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeaturesX; j++) {
                this.cachedGradient.set(i * numFeaturesX + j, gradientX.get(i).get(j));
            }
        }
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeaturesY; j++) {
                this.cachedGradient.set(i * numFeaturesY + j + offset, gradientY.get(i).get(j));
            }
        }

        this.flagGradientCached = true;
        return this.cachedGradient;
    }

    public Vector updateGradientX(int i) {
        int numData = this.X.getNumDataPoints();

        Vector firstTermX = this.X.getRow(i).minus(this.X0.getRow(i)).times(2 * this.alpha);
        Vector secondTermX = new DenseVector(this.X.getNumFeatures());
        for (int j = 0; j < numData; ++j) {
            if (i == j) {
                continue;
            }
            Vector diff = this.X.getRow(i).minus(this.X.getRow(j));
            double p_ij = (this.p[i][j] + this.p[j][i]) * 0.5 / (double)numData;
            double p_over_q_ij = Math.log(p_ij) - Math.log(this.q[i][j]);
            double scale = this.p[j][i] * (this.p_scale[i] - p_over_q_ij) * this.precision[i]
                    + this.p[i][j] * (this.p_scale[j] - p_over_q_ij) * this.precision[j];
            secondTermX = secondTermX.plus(diff.times(scale));
        }
        secondTermX = secondTermX.times(this.omega / (double)numData);
        Vector gradient = firstTermX.plus(secondTermX);

        return gradient;
    }

    public Vector updateGradientY(int i) {
        int numData = this.Y.getNumDataPoints();

        Vector firstTermY = this.Y.getRow(i).minus(this.Y0.getRow(i)).times(2 * this.beta);
        Vector secondTermY = new DenseVector(this.Y.getNumFeatures());
        Vector thirdTermY = new DenseVector(this.Y.getNumFeatures());
        for (int j = 0; j < numData; ++j) {
            if (i == j) {
                continue;
            }
            Vector diff = this.Y.getRow(i).minus(this.Y.getRow(j));
            Vector diff_proj = this.Y.getRow(i).minus(this.Y.getRow(j));
            Vector diff_user = this.U.getRow(i).minus(this.U.getRow(j));
            double scaleSecondTerm = diff_proj.dot(diff_proj) - diff_user.dot(diff_user);
            secondTermY = secondTermY.plus(diff.times(scaleSecondTerm));
            double p_ij = (this.p[i][j] + this.p[j][i]) * 0.5 / (double)numData;
            double scaleThirdTerm = (p_ij - this.q[i][j]) / (1 + diff_proj.dot(diff_proj));
            thirdTermY = thirdTermY.plus(diff.times(scaleThirdTerm));
        }
        secondTermY = secondTermY.times(8 * this.gamma);
        thirdTermY = thirdTermY.times(4 * this.omega);
        Vector gradient = firstTermY.plus(secondTermY).plus(thirdTermY);

        return gradient;
    }

    private void updateKLVariables() {
        int numData = this.X.getNumDataPoints();

        // denominator for p
        IntStream.range(0, numData).parallel().forEach(this::updateKLVariablePDenomi);

        // p[i][j] = p_i|j, non-symmetric
        IntStream.range(0, numData).parallel().forEach(this::updateKLVariableP);

        // denominator for q
        this.qdenomi = IntStream.range(0, numData).parallel().mapToDouble(this::updateKLVariableQDenomi).sum();

        // q[i][j] = q_ij, symmetric
        IntStream.range(0, numData).parallel().forEach(this::updateKLVariableQ);

        // log scale for p
        IntStream.range(0, numData).parallel().forEach(this::updateKLVariablePScale);

        this.flagKLVariableCached = true;
    }

    private void updateKLVariablePDenomi(int i) {
        int numData = this.X.getNumDataPoints();
        this.pdenomi[i] = 0.0;
        for (int j = 0; j < numData; ++j) {
            if (j == i) {
                continue;
            }
            Vector diff = this.X.getRow(j).minus(this.X.getRow(i));
            this.pdenomi[i] += Math.exp(-diff.dot(diff) * 0.5 * this.precision[i]);
        }
    }

    private void updateKLVariableP(int i) {
        int numData = this.X.getNumDataPoints();
        for (int j = 0; j < numData; ++j) {
            if (j == i) {
                continue;
            }
            Vector diff = this.X.getRow(j).minus(this.X.getRow(i));
            double diff_magnitude = diff.dot(diff);
            this.p[i][j] = Math.exp(-diff_magnitude * 0.5 * this.precision[j]) / this.pdenomi[j];
        }
    }

    private double updateKLVariableQDenomi(int i) {
        int numData = this.X.getNumDataPoints();
        double res = 0.0;
        for (int j = 0; j < numData; ++j) {
            if (j == i) {
                continue;
            }
            Vector diff = this.Y.getRow(j).minus(this.Y.getRow(i));
            double diff_magnitude = diff.dot(diff);
            res += 1.0 / (1 + diff_magnitude);
        }
        return res;
    }

    private void updateKLVariableQ(int i) {
        int numData = this.X.getNumDataPoints();
        for (int j = 0; j < numData; ++j) {
            if (j == i) {
                continue;
            }
            Vector diff = this.Y.getRow(j).minus(this.Y.getRow(i));
            double diff_magnitude = diff.dot(diff);
            this.q[i][j] = 1.0 / (1 + diff_magnitude);
            this.q[i][j] /= this.qdenomi;
        }
    }

    private void updateKLVariablePScale(int i) {
        int numData = this.X.getNumDataPoints();
        this.p_scale[i] = 0.0;
        for (int j = 0; j < numData; ++j) {
            if (j == i) {
                continue;
            }
            double p_ij = (this.p[j][i] + this.p[i][j]) * 0.5 / (double)numData;
            this.p_scale[i] += (Math.log(p_ij) - Math.log(this.q[i][j])) * this.p[j][i];
        }
    }
}
