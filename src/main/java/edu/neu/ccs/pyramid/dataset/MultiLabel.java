package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel implements Serializable{
    private static final long serialVersionUID = 2L;
    private Set<Integer> labels;


    public MultiLabel() {
        this.labels = new HashSet<>();
    }

    /**
     *
     * @param vector a binary label vector
     */
    public MultiLabel(Vector vector){
        this();
        for (Vector.Element element:vector.nonZeroes()){
            this.addLabel(element.index());
        }
    }

    public MultiLabel addLabel(int k) {
        this.labels.add(k);
        return this;
    }

    public void removeLabel(int k) {
        if (labels.contains(k)) {
            labels.remove(k);
        }
    }

    public void flipLabel(int k){
        if (labels.contains(k)){
            labels.remove(k);
        } else {
            labels.add(k);
        }
    }

    public boolean matchClass(int k){
        return labels.contains(k);
    }

    public Set<Integer> getMatchedLabels(){
        return labels;
    }

    public List<Integer> getMatchedLabelsOrdered(){
        return labels.stream().sorted().collect(Collectors.toList());
    }

    public static Set<Integer> union(MultiLabel multiLabel1, MultiLabel multiLabel2){
        Set<Integer> union = new HashSet<>();
        union.addAll(multiLabel1.getMatchedLabels());
        union.addAll(multiLabel2.getMatchedLabels());
        return union;
    }

    public static Set<Integer> intersection(MultiLabel multiLabel1, MultiLabel multiLabel2){
        Set<Integer> intersection = new HashSet<>();
        intersection.addAll(multiLabel1.getMatchedLabels());
        intersection.retainAll(multiLabel2.getMatchedLabels());
        return intersection;
    }


    public static Set<Integer> symmetricDifference(MultiLabel multiLabel1, MultiLabel multiLabel2){
        Set<Integer> union = union(multiLabel1,multiLabel2);
        Set<Integer> itersection = intersection(multiLabel1,multiLabel2);
        union.removeAll(itersection);
        return union;
    }

    public boolean outOfBound(int numClasses){
        for (int k:getMatchedLabels()){
            if (k> numClasses-1){
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return labels.stream().sorted().collect(Collectors.toList()).toString();
    }

    public String toStringWithExtLabels(LabelTranslator labelTranslator){
        return labels.stream().sorted().map(labelTranslator::toExtLabel).collect(Collectors.toList()).toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        MultiLabel that = (MultiLabel) o;

        if (!labels.equals(that.labels)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return labels.hashCode();
    }
}
