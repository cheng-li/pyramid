package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.FeatureRow;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/28/14.
 */
public class DecisionPath {
    private List<Node> path;
    private FeatureRow featureRow;

    public DecisionPath(RegressionTree tree, FeatureRow featureRow){
        this();
        this.featureRow = featureRow;
        int leafIndex = getMatchedLeaf(tree,featureRow);
        Node leaf = tree.leaves.get(leafIndex);
        this.path.add(leaf);
        Node tmp = leaf;
        while(tmp!=tree.root){
            tmp = tmp.getParent();
            this.path.add(tmp);
        }
    }

    DecisionPath() {
        this.path = new ArrayList<>();
    }

    //todo deal with probabilities
    private static int getMatchedLeaf(RegressionTree tree, FeatureRow featureRow){
        Vector vector = featureRow.getVector();
        for (int i=0;i<tree.getNumLeaves();i++){
            double prob = tree.probability(vector,tree.leaves.get(i));
            if (prob==1){
                return i;
            }
        }
        return 0;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i=this.path.size()-1;i>=0;i--){
            Node node = path.get(i);
            if (!node.isLeaf()){
                int featureIndex = node.getFeatureIndex();
                String featureName = node.getFeatureName();
                double threshold = node.getThreshold();
                double featureValue = this.featureRow.getVector().get(featureIndex);
                Node child = path.get(i-1);
                sb.append("feature ").append(featureIndex)
                        .append("(").append(featureName).append(")").append(" ");
                if (child==node.getLeftChild()){
                    sb.append(featureValue).append("<=").append(threshold).append(", ");
                } else {
                    sb.append(featureValue).append(">").append(threshold).append(", ");
                }
            } else {
                sb.append("score=").append(node.getValue());
            }

        }

        return sb.toString();
    }
}
