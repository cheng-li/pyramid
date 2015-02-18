package edu.neu.ccs.pyramid.regression.regression_tree;

import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by chengli on 2/18/15.
 */
public class Decision {
    private DecisionPath decisionPath;
    private double score;

    Decision() {
        this.decisionPath = new DecisionPath();
    }

    public Decision(RegressionTree tree, Vector vector){
        this();
        Node node = tree.getRoot();
        add(node,vector);
    }

    public DecisionPath getDecisionPath() {
        return decisionPath;
    }

    //todo deal with probabilities
    public void add (Node node, Vector vector){
        if (node.isLeaf()){
            this.score = node.getValue();
        } else {
            int featureIndex = node.getFeatureIndex();
            String featureName = node.getFeatureName();
            double threshold = node.getThreshold();
            double featureValue = vector.get(featureIndex);
            boolean direction = featureValue<=threshold;
            this.decisionPath.featureIndices.add(featureIndex);
            this.decisionPath.featureNames.add(featureName);
            this.decisionPath.thresholds.add(threshold);
            this.decisionPath.directions.add(direction);
            this.decisionPath.values.add(featureValue);
            Node child;
            if (direction){
                child = node.getLeftChild();
            } else {
                child = node.getRightChild();
            }
            add(child,vector);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(decisionPath.toString());
        sb.append("score = ").append(score);
        return sb.toString();
    }

    public static Decision merge(Decision decision1, Decision decision2){
        if (!decision1.getDecisionPath().equals(decision2.getDecisionPath())){
            throw new IllegalArgumentException("cannot merge decisions with different decision paths");
        }
        Decision decision = new Decision();
        decision.decisionPath = decision1.decisionPath.copy();
        decision.score = decision1.score + decision2.score;
        return decision;
    }

    public static List<Decision> merge(List<Decision> decisions){
        Map<DecisionPath,Double> map = new HashMap<>();
        for (Decision decision: decisions){
            double oldScore = map.getOrDefault(decision.decisionPath,0.0);
            double newScore = oldScore + decision.score;
            map.put(decision.decisionPath,newScore);
        }

        List<Decision> merged = new ArrayList<>();
        for (Map.Entry<DecisionPath,Double> entry: map.entrySet()){
            DecisionPath decisionPath = entry.getKey();
            double score = entry.getValue();
            Decision decision = new Decision();
            decision.decisionPath = decisionPath;
            decision.score = score;
            merged.add(decision);
        }
        return merged;
    }


}
