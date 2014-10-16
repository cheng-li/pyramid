package edu.neu.ccs.pyramid.regression.m_regression_tree;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
public class Node implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * output value of the node
     */
    private double value;
    /**
     * the feature for split
     */
    private int featureIndex;
    /**
     * the threshold for split
     */
    private double threshold;
    private Node leftChild;
    private Node rightChild;
    private double reduction;
    private boolean leaf;
    private Node parent;
    private boolean splitable;

    /**
     * for missing values
     * the probability of falling into the left child
     */
    private double leftProb;
    /**
     * for missing values
     * the probability of falling into the right child
     */
    private double rightProb;
    private String featureName = "unknown";

    /**
     * stores the indices of the data points on this node
     * just for training
     * will be cleaned later
     */
    private transient int[] dataAppearance;

    boolean isSplitable() {
        return splitable;
    }

    Node setSplitable(boolean splitable) {
        this.splitable = splitable;
        return this;
    }

    Node getLeftChild() {
        return leftChild;
    }

    Node setLeftChild(Node leftChild) {
        this.leftChild = leftChild;
        leftChild.parent = this;
        return this;
    }

    Node getRightChild() {
        return rightChild;
    }

    Node setRightChild(Node rightChild) {
        this.rightChild = rightChild;
        rightChild.parent = this;
        return this;
    }

    public double getReduction() {
        return reduction;
    }

    Node setReduction(double reduction) {
        this.reduction = reduction;
        return this;
    }

    boolean isLeaf() {
        return leaf;
    }

    boolean hasChildern(){
        return !isLeaf();
    }

    Node setLeaf(boolean leaf) {
        this.leaf = leaf;
        return this;
    }

    int[] getDataAppearance() {
        return dataAppearance;
    }

    Node setDataAppearance(int[] dataAppearance) {
        this.dataAppearance = dataAppearance;
        return this;
    }

    double getValue() {
        return value;
    }


    Node setValue(double value) {
        this.value = value;
        return this;
    }


    int getFeatureIndex() {
        return featureIndex;
    }

    Node setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
        return this;
    }

    double getThreshold() {
        return threshold;
    }

    Node setThreshold(double threshold) {
        this.threshold = threshold;
        return this;
    }

    Node getParent() {
        return parent;
    }


    String getFeatureName() {
        return featureName;
    }

    void setFeatureName(String featureName) {
        this.featureName = featureName;
    }

    /**
     * after split, free memory
     */
    void clearDataAppearance(){
        this.dataAppearance=null;
    }

    double getLeftProb() {
        return leftProb;
    }

    void setLeftProb(double leftProb) {
        this.leftProb = leftProb;
    }

    double getRightProb() {
        return rightProb;
    }

    void setRightProb(double rightProb) {
        this.rightProb = rightProb;
    }

    @Override
    public String toString() {
        return "Node{" +
                "leaf=" + leaf +
                ", splitable=" + splitable +
                ", leftProb=" + leftProb +
                ", rightProb=" + rightProb +
                ", featureName='" + featureName + '\'' +
                ", value=" + value +
                ", featureIndex=" + featureIndex +
                ", threshold=" + threshold +
                '}';
    }
}
