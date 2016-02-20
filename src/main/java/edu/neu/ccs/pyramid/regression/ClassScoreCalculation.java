package edu.neu.ccs.pyramid.regression;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 2/28/15.
 */
//todo json serialize
public class ClassScoreCalculation {
    private int internalClassIndex;
    private String className;
    private double classScore;
    private double classProbability;

    private List<Rule> rules;

    public ClassScoreCalculation(int internalClassIndex, String className, double totalScore) {
        this.internalClassIndex = internalClassIndex;
        this.className = className;
        this.classScore = totalScore;
        this.rules = new ArrayList<>();
    }


    public void addRule(Rule rule){
        this.rules.add(rule);
    }

    public double getClassScore() {
        return classScore;
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

    public double getClassProbability() {
        return classProbability;
    }

    public void setClassProbability(double classProbability) {
        this.classProbability = classProbability;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("ClassScoreCalculation{");
        sb.append("internalClassIndex=").append(internalClassIndex);
//        sb.append(", className='").append(className).append('\'');
        sb.append(", classScore=").append(classScore);
        sb.append(", classProbability=").append(classProbability);
        sb.append(", rules=").append(rules);
        sb.append('}');
        return sb.toString();
    }
}
