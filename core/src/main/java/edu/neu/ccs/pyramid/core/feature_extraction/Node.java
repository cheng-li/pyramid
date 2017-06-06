package edu.neu.ccs.pyramid.core.feature_extraction;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 3/28/15.
 */
public class Node<T> {
    T value;
    List<Node<T>> children;
    Node<T> parent;
    boolean leaf;
    int slop;
    int length;

    public Node() {
        this.children = new ArrayList<>();
    }

    public void addChild(Node<T> child){
        this.children.add(child);
        child.parent = this;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    public List<Node<T>> getChildren() {
        return children;
    }

    public void setChildren(List<Node<T>> children) {
        this.children = children;
        for (Node<T> node: children){
            node.parent = this;
        }
    }

    public Node<T> getParent() {
        return parent;
    }


    public boolean isLeaf() {
        return leaf;
    }

    public void setLeaf(boolean leaf) {
        this.leaf = leaf;
    }

    public int getSlop() {
        return slop;
    }

    public void setSlop(int slop) {
        this.slop = slop;
    }

    public int getLength() {
        return length;
    }

    public void setLength(int length) {
        this.length = length;
    }
}
