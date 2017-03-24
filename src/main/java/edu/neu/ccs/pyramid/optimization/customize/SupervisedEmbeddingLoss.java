package edu.neu.ccs.pyramid.optimization.customize;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by yuyuxu on 3/23/17.
 */
public class SupervisedEmbeddingLoss implements Optimizable.ByGradientValue {
    private DataSet transform;              // d x 2, transformation matrix
    private DataSet embedding;              // n x d, original embedding
    private DataSet projection;             // d x 2, projection provided by user
    private DataSet updatedEmbedding;       // n x d, updated embedding
    private DataSet updatedProjection;      // n x 2, updated projection computed from updated embedding
    private double alpha;
    private double beta;

    private boolean flagValueCached;
    private boolean flagGradientCached;
    private double cachedValue;
    private Vector cachedGradient;

    public SupervisedEmbeddingLoss(DataSet T, DataSet Q, DataSet P, double a, double b) {
        this.transform = T;
        this.embedding = Q;
        this.projection = P;
        this.updatedEmbedding = Q;
        this.updatedProjection = new DenseClfDataSet(Q.getNumDataPoints(), 2, false, 2);
        this.alpha = a;
        this.beta = b;

        this.flagValueCached = false;
        this.flagGradientCached = false;
        this.cachedGradient = new DenseVector(this.embedding.getNumDataPoints() * this.embedding.getNumFeatures());

        this.updateProjection(Q);

        System.out.println("SupervisedEmbeddingLoss loss function initialized ...");
    }

    private void updateProjection(DataSet Q) {
        int numData = Q.getNumDataPoints();
        for (int i = 0; i < numData; i++) {
            Vector q_i = this.updatedEmbedding.getRow(i);
            this.updatedProjection.setFeatureValue(i, 0, this.transform.getColumn(0).dot(q_i));
            this.updatedProjection.setFeatureValue(i, 1, this.transform.getColumn(1).dot(q_i));
        }
    }

    public DataSet getUpdatedEmbedding() {
        return this.updatedEmbedding;
    }

    public DataSet getUpdatedProjection() {
        return this.updatedProjection;
    }

    public Vector getParameters() {
        int numData = this.updatedEmbedding.getNumDataPoints();
        int numFeatures = this.updatedEmbedding.getNumFeatures();

        Vector pVec = new DenseVector(numData * numFeatures);
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                pVec.set(i * numFeatures + j, this.updatedEmbedding.getRow(i).get(j));
            }
        }
        return pVec;
    }

    public void setParameters(Vector parameters) {
        int numData = this.updatedEmbedding.getNumDataPoints();
        int numFeatures = this.updatedEmbedding.getNumFeatures();

        // update embedding
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                this.updatedEmbedding.setFeatureValue(i, j, parameters.get(i * numFeatures + j));
            }
        }

        // update projection
        this.updateProjection(this.updatedEmbedding);

        // reset flag
        this.flagValueCached = false;
        this.flagGradientCached = false;
    }

    public double getValue() {
        if (this.flagValueCached) {
            return this.cachedValue;
        }

        int numData = this.updatedEmbedding.getNumDataPoints();
        this.cachedValue =  IntStream.range(0, numData)
                .parallel()
                .mapToDouble(this::getValue).sum();
        this.flagValueCached = true;
        return this.cachedValue;
    }

    private double getValue(int i) {
        int numData = this.updatedEmbedding.getNumDataPoints();

        double loss = 0.0;
        Vector q_i = this.updatedEmbedding.getRow(i);
        Vector q_i_orig = this.embedding.getRow(i);
        loss += this.alpha * q_i.getDistanceSquared(q_i_orig);

        double pi_x = this.updatedProjection.getRow(i).get(0);
        double pi_y = this.updatedProjection.getRow(i).get(1);
        double user_pi_x = this.projection.getRow(i).get(0);
        double user_pi_y = this.projection.getRow(i).get(1);
        for (int j = 0; j < numData; j++) {
            double pj_x = this.updatedProjection.getRow(j).get(0);
            double pj_y = this.updatedProjection.getRow(j).get(1);
            double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);

            double user_pj_x = this.projection.getRow(j).get(0);
            double user_pj_y = this.projection.getRow(j).get(1);
            double d_sq = (user_pi_x - user_pj_x) * (user_pi_x - user_pj_x) +
                    (user_pi_y - user_pj_y) * (user_pi_y - user_pj_y);

            loss += this.beta * (p_sq - d_sq) * (p_sq - d_sq);
        }
        return loss;
    }

    public Vector getGradient() {
        if (this.flagGradientCached) {
            return this.cachedGradient;
        }

        int numData = this.updatedEmbedding.getNumDataPoints();
        int numFeatures = this.updatedEmbedding.getNumFeatures();

        List<Vector> gradients = IntStream.range(0, numData)
                .parallel()
                .mapToObj(this::getGradient).collect(Collectors.toList());
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                this.cachedGradient.set(i * numFeatures + j, gradients.get(i).get(j));
            }
        }
        this.flagGradientCached = true;
        return this.cachedGradient;
    }

    private Vector getGradient(int i) {
        int numData = this.updatedEmbedding.getNumDataPoints();
        int numFeatures = this.updatedEmbedding.getNumFeatures();

        Vector gradient = new DenseVector(numFeatures);
        Vector q_i = this.updatedEmbedding.getRow(i);
        Vector q_i_orig = this.embedding.getRow(i);
        for (int j = 0; j < gradient.size(); j++) {
            gradient.set(j, gradient.get(j) + (q_i.get(j) - q_i_orig.get(j)) * 2.0 * this.alpha);
        }

        double pi_x = this.updatedProjection.getRow(i).get(0);
        double pi_y = this.updatedProjection.getRow(i).get(1);
        double user_pi_x = this.projection.getRow(i).get(0);
        double user_pi_y = this.projection.getRow(i).get(1);
        Vector p_dist_vec = new DenseVector(2);
        for (int j = 0; j < numData; j++) {
            double pj_x = this.updatedProjection.getRow(j).get(0);
            double pj_y = this.updatedProjection.getRow(j).get(1);
            double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);

            double user_pj_x = this.projection.getRow(j).get(0);
            double user_pj_y = this.projection.getRow(j).get(1);
            double d_sq = (user_pi_x - user_pj_x) * (user_pi_x - user_pj_x) +
                    (user_pi_y - user_pj_y) * (user_pi_y - user_pj_y);

            double scale = 4.0 * this.beta * (p_sq - d_sq);
            p_dist_vec.set(0, p_dist_vec.get(0) + (pi_x - pj_x) * scale);
            p_dist_vec.set(1, p_dist_vec.get(1) + (pi_y - pj_y) * scale);
        }

        for (int j = 0; j < gradient.size(); j++) {
            gradient.set(j, this.transform.getRow(j).dot(p_dist_vec) + gradient.get(j));
        }
        return gradient;
    }
}