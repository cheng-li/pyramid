package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.regression.Rule;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by chengli on 2/18/15.
 */
public class TreeRule implements Rule {
    private Checks checks;
    private double score;

    TreeRule() {
        this.checks = new Checks();
    }

    public TreeRule(RegressionTree tree, Vector vector){
        this();
        Node node = tree.getRoot();
        add(node,vector);
    }

    public Checks getChecks() {
        return checks;
    }

    public double getScore() {
        return score;
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
            this.checks.featureIndices.add(featureIndex);
            this.checks.featureNames.add(featureName);
            this.checks.thresholds.add(threshold);
            this.checks.directions.add(direction);
            this.checks.values.add(featureValue);
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
        sb.append(checks.toString());
        sb.append("score = ").append(score);
        return sb.toString();
    }

    public static TreeRule merge(TreeRule treeRule1, TreeRule treeRule2){
        if (!treeRule1.getChecks().equals(treeRule2.getChecks())){
            throw new IllegalArgumentException("cannot merge decisions with different decision paths");
        }
        TreeRule treeRule = new TreeRule();
        treeRule.checks = treeRule1.checks.copy();
        treeRule.score = treeRule1.score + treeRule2.score;
        return treeRule;
    }

    public static List<TreeRule> merge(List<TreeRule> treeRules){
        Map<Checks,Double> map = new HashMap<>();
        for (TreeRule treeRule : treeRules){
            double oldScore = map.getOrDefault(treeRule.checks,0.0);
            double newScore = oldScore + treeRule.score;
            map.put(treeRule.checks,newScore);
        }

        List<TreeRule> merged = new ArrayList<>();
        for (Map.Entry<Checks,Double> entry: map.entrySet()){
            Checks checks = entry.getKey();
            double score = entry.getValue();
            TreeRule treeRule = new TreeRule();
            treeRule.checks = checks;
            treeRule.score = score;
            merged.add(treeRule);
        }
        return merged;
    }


}
