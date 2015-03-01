package edu.neu.ccs.pyramid.regression;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 2/28/15.
 */

public class ClassScoreCalculation {
    private int internalClassIndex;
    private String className;
    private double totalScore;

    private List<Rule> rules;

    public ClassScoreCalculation(int internalClassIndex, String className, double totalScore) {
        this.internalClassIndex = internalClassIndex;
        this.className = className;
        this.totalScore = totalScore;
        this.rules = new ArrayList<>();
    }


    public void addRule(Rule rule){
        this.rules.add(rule);
    }

    public double getTotalScore() {
        return totalScore;
    }

    public List<Rule> getRules() {
        return rules;
    }

    public int getInternalClassIndex() {
        return internalClassIndex;
    }

    public String getClassName() {
        return className;
    }
}
