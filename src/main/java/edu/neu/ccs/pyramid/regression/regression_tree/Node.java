package edu.neu.ccs.pyramid.regression.regression_tree;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
class Node implements Serializable {

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
    private transient double reduction;
    private boolean leaf;
    private Node parent;
    private boolean splitable;

    /**
     * stores the indices of the data points on this node
     * just for training
     * will be cleaned later
     */
    private transient int[] dataAppearance;

    boolean isSplitable() {
        return splitable;
    }

    void setSplitable(boolean splitable) {
        this.splitable = splitable;
    }

    Node getLeftChild() {
        return leftChild;
    }

    void setLeftChild(Node leftChild) {
        this.leftChild = leftChild;
    }

    Node getRightChild() {
        return rightChild;
    }

    void setRightChild(Node rightChild) {
        this.rightChild = rightChild;
    }

    double getReduction() {
        return reduction;
    }

    void setReduction(double reduction) {
        this.reduction = reduction;
    }

    boolean isLeaf() {
        return leaf;
    }

    boolean hasChildern(){
        return !isLeaf();
    }

    void setLeaf(boolean leaf) {
        this.leaf = leaf;
    }

    int[] getDataAppearance() {
        return dataAppearance;
    }

    void setDataAppearance(int[] dataAppearance) {
        this.dataAppearance = dataAppearance;
    }

    double getValue() {
        return value;
    }


    void setValue(double value) {
        this.value = value;
    }


    int getFeatureIndex() {
        return featureIndex;
    }

    void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }

    double getThreshold() {
        return threshold;
    }

    void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    Node getParent() {
        return parent;
    }

    void setParent(Node parent) {
        this.parent = parent;
    }

    /**
     * after split, free memory
     */
    void clearDataAppearance(){
        this.dataAppearance=null;
    }

    @Override
    public String toString() {
        return "Node{" +
                "value=" + value +
                ", featureIndex=" + featureIndex +
                ", threshold=" + threshold +
                ", leaf=" + leaf +
                '}';
    }
}
