package edu.neu.ccs.pyramid.classification;

/**
 * Created by chengli on 2/28/15.
 */
public class ClassProbability {
    private int internalClassIndex;
    private String className;
    private double probability;

    public ClassProbability(int internalClassIndex, String className, double probability) {
        this.internalClassIndex = internalClassIndex;
        this.className = className;
        this.probability = probability;
    }

    public int getInternalClassIndex() {
        return internalClassIndex;
    }

    public String getClassName() {
        return className;
    }

    public double getProbability() {
        return probability;
    }
}
