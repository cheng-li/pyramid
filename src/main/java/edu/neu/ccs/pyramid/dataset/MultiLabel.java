package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel implements Serializable{
    private static final long serialVersionUID = 1L;
    private Set<Integer> labels;
    private boolean[] labelsVector;

    public MultiLabel(int numClasses) {
        this.labels = new HashSet<>();
        this.labelsVector= new boolean[numClasses];
    }

    public MultiLabel addLabel(int k){
        this.labels.add(k);
        this.labelsVector[k]=true;
        return this;
    }

    public boolean matchClass(int k){
        return labelsVector[k];
    }

    public Set<Integer> getMatchedLabels(){
        return labels;
    }

    public static boolean equivalent(MultiLabel multiLabel1, MultiLabel multiLabel2){
        if (multiLabel1.labelsVector.length!=multiLabel2.labelsVector.length){
            return false;
        }

        for (int i=0;i<multiLabel1.labelsVector.length;i++){
            if (multiLabel1.labelsVector[i]!=multiLabel2.labelsVector[i]){
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return  "{"+
                "labels=" + labels +
                ", labelsVector=" + Arrays.toString(labelsVector)
                +"}";
    }
}
