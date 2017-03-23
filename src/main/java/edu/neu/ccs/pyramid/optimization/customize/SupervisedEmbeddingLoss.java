package edu.neu.ccs.pyramid.optimization.customize;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by yuyuxu on 3/23/17.
 */
public class SupervisedEmbeddingLoss implements Optimizable.ByGradientValue {
    private static final Logger logger = LogManager.getLogger();
    private DataSet distance;               // n x n
    private DataSet transform;              // d x 2
    private DataSet embedding;              // n x d
    private DataSet updatedEmbedding;       // n x d
    private DataSet updatedProjection;      // n x 2
    private Double alpha;
    private Double beta;

    public SupervisedEmbeddingLoss(DataSet T, DataSet Q, DataSet P, Double a, Double b) {
        this.distance = new DenseClfDataSet(Q.getNumDataPoints(), Q.getNumDataPoints(), false, 2);
        this.transform = T;
        this.embedding = Q;
        this.updatedEmbedding = Q;
        this.updatedProjection = new DenseClfDataSet(Q.getNumDataPoints(), 2, false, 2);
        this.alpha = a;
        this.beta = b;


        int numData = Q.getNumDataPoints();
        for (int i = 0; i < numData; i++) {
            double pi_x = P.getRow(i).get(0);
            double pi_y = P.getRow(i).get(1);
            for (int j = 0; j < numData; j++) {
                double pj_x = P.getRow(j).get(0);
                double pj_y = P.getRow(j).get(1);
                double d = Math.sqrt((pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y));
                this.distance.setFeatureValue(i, j, d);
            }
        }
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
        int vecSize = numData * numFeatures;
        Vector pVec = new DenseVector(vecSize);
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
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                this.updatedEmbedding.setFeatureValue(i, j, parameters.get(i * numFeatures + j));
            }
        }
        this.updateProjection(this.updatedEmbedding);
    }

    public double getValue() {
        Double loss = 0.0;
        int numData = this.updatedEmbedding.getNumDataPoints();
        for (int i = 0; i < numData; i++) {
            Vector q_i = this.updatedEmbedding.getRow(i);
            Vector q_i_orig = this.embedding.getRow(i);
            loss += this.alpha * q_i.getDistanceSquared(q_i_orig);
            for (int j = 0; j < numData; j++) {
                Vector q_j = this.updatedEmbedding.getRow(j);
                double pi_x = this.transform.getColumn(0).dot(q_i);
                double pi_y = this.transform.getColumn(1).dot(q_i);
                double pj_x = this.transform.getColumn(0).dot(q_j);
                double pj_y = this.transform.getColumn(1).dot(q_j);
                double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);
                double d_sq = this.distance.getRow(i).get(j) * this.distance.getRow(i).get(j);
                loss += this.beta * (p_sq - d_sq) * (p_sq - d_sq);
            }
        }
        return loss;
    }

    public Vector getGradient() {
        int numData = this.updatedEmbedding.getNumDataPoints();
        int numFeatures = this.updatedEmbedding.getNumFeatures();
        int vecSize = numData * numFeatures;
        Vector finalGradient = new DenseVector(vecSize);

        for (int i = 0; i < numData; i++) {
            Vector gradient = new DenseVector(numFeatures);
            Vector q_i = this.updatedEmbedding.getRow(i);
            Vector q_i_orig = this.embedding.getRow(i);
            gradient = gradient.plus(q_i.minus(q_i_orig).times(2.0 * this.alpha));

            for (int j = 0; j < numData; j++) {
                Vector q_j = this.updatedEmbedding.getRow(j);
                double pi_x = this.transform.getColumn(0).dot(q_i);
                double pi_y = this.transform.getColumn(1).dot(q_i);
                double pj_x = this.transform.getColumn(0).dot(q_j);
                double pj_y = this.transform.getColumn(1).dot(q_j);
                double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);
                double d_sq = this.distance.getRow(i).get(j) * this.distance.getRow(i).get(j);
                Vector p_dist_vec = new DenseVector(2);
                p_dist_vec.set(0, pi_x - pj_x);
                p_dist_vec.set(1, pi_y - pj_y);
                Vector tmp = new DenseVector(this.transform.getNumDataPoints());
                for (int k = 0; k < this.transform.getNumDataPoints(); k++) {
                    tmp.set(k, this.transform.getRow(k).dot(p_dist_vec));
                }
                gradient = gradient.plus(tmp.times(4.0 * this.beta * (p_sq - d_sq)));
            }

            for (int j = 0; j < numFeatures; j++) {
                finalGradient.set(i * numFeatures + j, gradient.get(j));
            }
        }
        return finalGradient;
    }
}