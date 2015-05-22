package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/11/14.
 */
public class LabelTranslator implements Serializable{
    private static final long serialVersionUID = 1L;
    private Map<Integer,String> intToExt;
    private Map<String, Integer> extToInt;

    /**
     * use user specified orders
     * @param extLabels
     */
    public LabelTranslator(List<String> extLabels){
        this();
        for (int i=0;i<extLabels.size();i++){
            String extLabel = extLabels.get(i);
            this.intToExt.put(i,extLabel);
            this.extToInt.put(extLabel,i);
        }
    }

    /**
     * use user specified orders
     * @param extLabels
     */
    public LabelTranslator(String[] extLabels){
        this();
        for (int i=0;i<extLabels.length;i++){
            String extLabel = extLabels[i];
            this.intToExt.put(i,extLabel);
            this.extToInt.put(extLabel,i);
        }
    }

    /**
     * use user specified orders
     * @param intToExtMap
     */
    public LabelTranslator(Map<Integer, String> intToExtMap){
        this();
        for (Map.Entry<Integer, String> entry:intToExtMap.entrySet()){
            Integer intLabel = entry.getKey();
            String extLabel = entry.getValue();
            this.intToExt.put(intLabel,extLabel);
            this.extToInt.put(extLabel,intLabel);
        }
    }

    /**
     * int labels are assigned based on the alphabetical order of ext labels
     * @param extLabels
     */
    public LabelTranslator(Set<String> extLabels){
        this();
        List<String> extLabelList = extLabels.stream().sorted()
                .collect(Collectors.toList());
        for (int i=0;i<extLabelList.size();i++){
            String extLabel = extLabelList.get(i);
            this.intToExt.put(i,extLabel);
            this.extToInt.put(extLabel,i);
        }
    }


    public String toExtLabel(int intLabel){
        return intToExt.get(intLabel);
    }

    public int toIntLabel(String extLabel){
        if (extToInt==null){
            System.out.println("BUG: extToInt is null");
        }
        return extToInt.get(extLabel);
    }

    private LabelTranslator() {
        this.intToExt = new HashMap<>();
        this.extToInt = new HashMap<>();
    }



    /**
     * int labels are assigned based on the alphabetical order of ext labels
     * @return
     */
    public static Builder getBuilder(){
        return new Builder();
    }

    public int getNumClasses(){
        return this.intToExt.size();
    }

    @Override
    public String toString() {
        return "LabelTranslator{" +
                "intToExt=" + intToExt +
                ", extToInt=" + extToInt +
                '}';
    }

    public static class Builder{

        Builder() {
            this.extLabels = new HashSet<>();
        }

        private Set<String> extLabels;

        public Builder addExtLabel(String extLabel){
            this.extLabels.add(extLabel);
            return this;
        }

        public Builder addExtLabels(Collection<String> extLabels){
            this.extLabels.addAll(extLabels);
            return this;
        }

        public LabelTranslator build(){
            LabelTranslator translator = new LabelTranslator();
            List<String> extLabelList = this.extLabels.stream().sorted()
                    .collect(Collectors.toList());
            for (int i=0;i<extLabelList.size();i++){
                String extLabel = extLabelList.get(i);
                translator.intToExt.put(i,extLabel);
                translator.extToInt.put(extLabel,i);
            }
            return translator;
        }
    }

    public static LabelTranslator newDefaultLabelTranslator(int numClasses){
        List<String> extLabels = IntStream.range(0,numClasses)
                .mapToObj(i -> ""+i).collect(Collectors.toList());
        return new LabelTranslator(extLabels);
    }
}
