package edu.neu.ccs.pyramid.core.feature_extraction;

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
        tree.root.setSlop(0);
        tree.root.setLength(1);
        grow(tree,tree.root,n,slop);
        this.positionTemplate = tree.getAllPaths();
    }

    private static void grow(KaryTree<Integer> tree, Node<Integer> node, int n, int slop){
        int currentN= node.getLength();
        int currentSlop = node.getSlop();
        if (currentN==n){
            tree.leaves.add(node);
            return;
        }
        for (int i=0;i<=slop-currentSlop;i++){
            Node<Integer> child = new Node<>();
            child.setValue(node.getValue() + i+1);
            child.setLength(currentN+1);
            child.setSlop(currentSlop+i);
            node.addChild(child);
            grow(tree,child,n,slop);
        }
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
