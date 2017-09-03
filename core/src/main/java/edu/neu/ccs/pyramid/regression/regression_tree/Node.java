package edu.neu.ccs.pyramid.regression.regression_tree;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
public class Node implements Serializable {

    private static final long serialVersionUID = 2L;
    /**
     * id of the node
     */
    private int id;

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


    //todo this should be transient? maybe doesn't matter as it is cleaned
    private double[] probs;

    boolean isSplitable() {
        return splitable;
    }

    Node setSplitable(boolean splitable) {
        this.splitable = splitable;
        return this;
    }

    public Node getLeftChild() {
        return leftChild;
    }

    Node setLeftChild(Node leftChild) {
        this.leftChild = leftChild;
        leftChild.parent = this;
        return this;
    }

    public Node getRightChild() {
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

    double[] getProbs() {
        return probs;
    }

    Node setProbs(double[] probs) {
        this.probs = probs;
        return this;
    }

    public double getValue() {
        return value;
    }


    Node setValue(double value) {
        this.value = value;
        return this;
    }


    public int getFeatureIndex() {
        return featureIndex;
    }

    Node setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
        return this;
    }

    public double getThreshold() {
        return threshold;
    }

    Node setThreshold(double threshold) {
        this.threshold = threshold;
        return this;
    }

    Node getParent() {
        return parent;
    }



    /**
     * after split, free memory
     */
    void clearProbs(){
        this.probs=null;
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

    public int getId() {
        return id;
    }

    public Node setId(int id) {
        this.id = id;
        return this;
    }

    @Override
    public String toString() {
        return "Node{" +
                "leaf=" + leaf +
                ", splitable=" + splitable +
                ", leftProb=" + leftProb +
                ", rightProb=" + rightProb +
                ", value=" + value +
                ", featureIndex=" + featureIndex +
                ", threshold=" + threshold +
                '}';
    }
}
