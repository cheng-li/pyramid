package edu.neu.ccs.pyramid.optimization;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by yuyuxu on 3/7/17.
 */
public class SupervisedEmbeddingLoss implements Optimizable.ByGradientValue {
    private static final Logger logger = LogManager.getLogger();
    private DataSet distMatrix; // n x n
    private DataSet projMatrix; // d x 2
    private DataSet embeddingMatrix; // n x d
    private DataSet updatedEmbeddingMatrix; // n x d
    private Double alpha;
    private Double beta;

    public SupervisedEmbeddingLoss(DataSet D, DataSet P, DataSet Q, Double a, Double b) {
        this.distMatrix = D;
        this.projMatrix = P;
        this.embeddingMatrix = Q;
        this.updatedEmbeddingMatrix = Q;
        this.alpha = a;
        this.beta = b;

        System.out.println("Init SupervisedEmbeddingLoss loss function ...");
    }

    public DataSet getUpdatedEmbeddingMatrix() {
        return this.updatedEmbeddingMatrix;
    }

    public Vector getParameters() {
        int numData = this.updatedEmbeddingMatrix.getNumDataPoints();
        int numFeatures = this.updatedEmbeddingMatrix.getNumFeatures();
        int vecSize = numData * numFeatures;
        Vector pVec = new DenseVector(vecSize);
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                pVec.set(i * numFeatures + j, this.updatedEmbeddingMatrix.getRow(i).get(j));
            }
        }
        return pVec;
    }


    public void setParameters(Vector parameters) {
        int numData = this.updatedEmbeddingMatrix.getNumDataPoints();
        int numFeatures = this.updatedEmbeddingMatrix.getNumFeatures();
        for (int i = 0; i < numData; i++) {
            for (int j = 0; j < numFeatures; j++) {
                this.updatedEmbeddingMatrix.setFeatureValue(i, j, parameters.get(i * numFeatures + j));
            }
        }
    }


    public double getValue() {
        Double loss = 0.0;
        int numData = this.updatedEmbeddingMatrix.getNumDataPoints();
        for (int i = 0; i < numData; i++) {
            Vector q_i = this.updatedEmbeddingMatrix.getRow(i);
            Vector q_i_orig = this.embeddingMatrix.getRow(i);
            loss += this.alpha * q_i.getDistanceSquared(q_i_orig);
            for (int j = 0; j < numData; j++) {
                Vector q_j = this.updatedEmbeddingMatrix.getRow(j);
                double pi_x = this.projMatrix.getColumn(0).dot(q_i);
                double pi_y = this.projMatrix.getColumn(1).dot(q_i);
                double pj_x = this.projMatrix.getColumn(0).dot(q_j);
                double pj_y = this.projMatrix.getColumn(1).dot(q_j);
                double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);
                double d_sq = this.distMatrix.getRow(i).get(j) * this.distMatrix.getRow(i).get(j);
                loss += this.beta * (p_sq - d_sq) * (p_sq - d_sq);
            }
        }
        return loss;
    }


    public Vector getGradient() {
        int numData = this.updatedEmbeddingMatrix.getNumDataPoints();
        int numFeatures = this.updatedEmbeddingMatrix.getNumFeatures();
        int vecSize = numData * numFeatures;
        Vector finalGradient = new DenseVector(vecSize);

        for (int i = 0; i < numData; i++) {
            Vector gradient = new DenseVector(numFeatures);
            Vector q_i = this.updatedEmbeddingMatrix.getRow(i);
            Vector q_i_orig = this.embeddingMatrix.getRow(i);
            gradient = gradient.plus(q_i.minus(q_i_orig).times(2.0 * this.alpha));

            for (int j = 0; j < numData; j++) {
                Vector q_j = this.updatedEmbeddingMatrix.getRow(j);
                double pi_x = this.projMatrix.getColumn(0).dot(q_i);
                double pi_y = this.projMatrix.getColumn(1).dot(q_i);
                double pj_x = this.projMatrix.getColumn(0).dot(q_j);
                double pj_y = this.projMatrix.getColumn(1).dot(q_j);
                double p_sq = (pi_x - pj_x) * (pi_x - pj_x) + (pi_y - pj_y) * (pi_y - pj_y);
                double d_sq = this.distMatrix.getRow(i).get(j) * this.distMatrix.getRow(i).get(j);
                Vector p_dist_vec = new DenseVector(2);
                p_dist_vec.set(0, pi_x - pj_x);
                p_dist_vec.set(1, pi_y - pj_y);
                Vector tmp = new DenseVector(this.projMatrix.getNumDataPoints());
                for (int k = 0; k < this.projMatrix.getNumDataPoints(); k++) {
                    tmp.set(k, this.projMatrix.getRow(k).dot(p_dist_vec));
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
