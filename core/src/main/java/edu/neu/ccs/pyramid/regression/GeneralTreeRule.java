package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.regression.regression_tree.GeneralChecks;
import edu.neu.ccs.pyramid.regression.regression_tree.Node;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.regression.regression_tree.TreeRule;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GeneralTreeRule {
    private GeneralChecks checks;
    private double score;


    private GeneralTreeRule(){
    }

    public GeneralTreeRule(RegressionTree tree, Node leaf){
        this.checks = new GeneralChecks(tree, leaf);
        this.score = leaf.getValue();
    }

    public GeneralChecks getGeneralChecks() {
        return checks;
    }

    public double getScore() {
        return score;
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(checks.toString());
        sb.append("score = ").append(score);
        return sb.toString();
    }

    public static GeneralTreeRule merge(GeneralTreeRule treeRule1, GeneralTreeRule treeRule2){
        if (!treeRule1.getGeneralChecks().equals(treeRule2.getGeneralChecks())){
            throw new IllegalArgumentException("cannot merge decisions with different decision paths");
        }
        GeneralTreeRule treeRule = new GeneralTreeRule();
        treeRule.checks = treeRule1.checks.copy();
        treeRule.score = treeRule1.score + treeRule2.score;
        return treeRule;
    }

    public static List<GeneralTreeRule> merge(List<GeneralTreeRule> treeRules){
        Map<GeneralChecks,Double> map = new HashMap<>();
        for (GeneralTreeRule treeRule : treeRules){
            double oldScore = map.getOrDefault(treeRule.checks,0.0);
            double newScore = oldScore + treeRule.score;
            map.put(treeRule.checks,newScore);
        }

        List<GeneralTreeRule> merged = new ArrayList<>();
        for (Map.Entry<GeneralChecks,Double> entry: map.entrySet()){
            GeneralChecks checks = entry.getKey();
            double score = entry.getValue();
            GeneralTreeRule treeRule = new GeneralTreeRule();
            treeRule.checks = checks;
            treeRule.score = score;
            merged.add(treeRule);
        }
        return merged;
    }
}
