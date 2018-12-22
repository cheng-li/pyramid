package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.ListUtil;
import edu.neu.ccs.pyramid.util.PrintUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel implements Serializable{
    private static final long serialVersionUID = 3L;
    private BitSet labels;


    public MultiLabel() {
        this.labels = new BitSet();
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


    public MultiLabel copy(){
        MultiLabel c= new MultiLabel();
        for (int i = labels.nextSetBit(0); i >= 0; i = labels.nextSetBit(i+1)) {
            c.addLabel(i);
        }
        return c;
    }

    /**
     * return binary vector
     * @param length
     * @return
     */
    public Vector toVector(int length){
        Vector vector = new DenseVector(length);
        for (int i = labels.nextSetBit(0); i >= 0; i = labels.nextSetBit(i+1)){
            vector.set(i,1);
        }
        return vector;
    }

    public MultiLabel addLabel(int k) {
        this.labels.set(k);
        return this;
    }

    public void removeLabel(int k) {
        this.labels.clear(k);
    }

    public void removeAllLabels(){
        for (int l: getMatchedLabels()){
            removeLabel(l);
        }
    }

    public void flipLabel(int k){
        this.labels.flip(k);
    }

    public boolean matchClass(int k){
        return labels.get(k);
    }

    public Set<Integer> getMatchedLabels(){
        Set<Integer> set = new HashSet<>();
        for (int i = labels.nextSetBit(0); i >= 0; i = labels.nextSetBit(i+1)) {
            set.add(i);
        }
        return set;
    }

    public int getNumMatchedLabels(){
        return labels.cardinality();
    }

    public List<Integer> getMatchedLabelsOrdered(){
        return getMatchedLabels().stream().sorted().collect(Collectors.toList());
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
        Set<Integer> intersection = intersection(multiLabel1,multiLabel2);
        union.removeAll(intersection);
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

    //todo test
    public boolean isSubsetOf(MultiLabel superSet){
        for (int i = labels.nextSetBit(0); i >= 0; i = labels.nextSetBit(i+1)) {
            if (!superSet.matchClass(i)){
                return false;
            }
        }
        return true;
    }

    public static boolean agreeOnLabel(MultiLabel multiLabel1, MultiLabel multiLabel2, int labelIndex){
        if (multiLabel1.matchClass(labelIndex)&&multiLabel2.matchClass(labelIndex)){
            return true;
        }


        if ((!multiLabel1.matchClass(labelIndex))&&(!multiLabel2.matchClass(labelIndex))){
            return true;
        }

        return false;

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append(ListUtil.toSimpleString(getMatchedLabels().stream().sorted().collect(Collectors.toList())));
        sb.append("}");
        return sb.toString();
    }

    public String toSimpleString() {
        StringBuilder sb = new StringBuilder();
        sb.append(ListUtil.toSimpleString(getMatchedLabels().stream().sorted().collect(Collectors.toList())));
        return sb.toString();
    }

    public String toStringWithExtLabels(LabelTranslator labelTranslator){
        return getMatchedLabels().stream().sorted().map(labelTranslator::toExtLabel).collect(Collectors.toList()).toString();
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
