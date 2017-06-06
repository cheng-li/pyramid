package edu.neu.ccs.pyramid.core.feature_extraction;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 3/28/15.
 */
public class KaryTree<T> {
    public Node<T> root;
    public List<Node<T>> leaves;

    public KaryTree() {
        root = new Node<>();
        leaves = new ArrayList<>();
    }

    public List<List<T>> getAllPaths(){
        return leaves.stream().map(this::getPath).collect(Collectors.toList());
    }

    public List<T> getPath(Node<T> leaf){
        List<T> list = new ArrayList<>();
        Node<T> node = leaf;
        while(true){
            list.add(node.getValue());
            if (node==root){
                break;
            }
            node = node.getParent();
        }
        Collections.reverse(list);
        return list;
    }

}
