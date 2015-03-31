package edu.neu.ccs.pyramid.feature_extraction;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 3/28/15.
 */
public class NgramTemplate {
    private int n;
    private int slop;
    private String field;
    private List<List<Integer>> positionTemplate;

    public NgramTemplate(String field, int n, int slop) {
        this.field = field;
        this.n = n;
        this.slop = slop;
        KaryTree<Integer> tree = new KaryTree<>();
        tree.root.setValue(0);
        List<Node<Integer>> currentLeaves = new ArrayList<>();
        currentLeaves.add(tree.root);
        for (int i=0;i<n-1;i++){
            List<Node<Integer>> nextLeaves = new ArrayList<>();
            for (Node<Integer> leaf: currentLeaves){
                for (int j=0;j<=slop;j++){
                    Node<Integer> child = new Node<>();
                    child.setValue(leaf.getValue() + j+1);
                    leaf.addChild(child);
                    nextLeaves.add(child);
                }
            }
            currentLeaves = nextLeaves;
        }
        tree.leaves = currentLeaves;
        this.positionTemplate = tree.getAllPaths();
    }

    public int getN() {
        return n;
    }

    public int getSlop() {
        return slop;
    }

    public String getField() {
        return field;
    }

    public List<List<Integer>> getPositionTemplate() {
        return positionTemplate;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("NgramTemplate{");
        sb.append("n=").append(n);
        sb.append(", slop=").append(slop);
        sb.append(", positionTemplate=").append(positionTemplate);
        sb.append('}');
        return sb.toString();
    }
}
