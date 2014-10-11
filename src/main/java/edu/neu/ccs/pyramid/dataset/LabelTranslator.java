package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 10/11/14.
 */
public class LabelTranslator implements Serializable{
    private static final long serialVersionUID = 1L;
    private Map<Integer,String> intToExt;
    private Map<String, Integer> extToInt;

    public String toExtLabel(int intLabel){
        return intToExt.get(intLabel);
    }

    public int toIntLabel(String extLabel){
        return extToInt.get(extLabel);
    }

    private LabelTranslator() {
        this.intToExt = new HashMap<>();
        this.extToInt = new HashMap<>();
    }

    public static Builder getBuilder(){
        return new Builder();
    }

    public int getClasses(){
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
}
