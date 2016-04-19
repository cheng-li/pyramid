package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 3/25/16.
 */
public class Enumerator {

    public static List<MultiLabel> enumerate(int numClasses){
        List<MultiLabel> list = new ArrayList<>();
        int size = (int)Math.pow(2,numClasses);
        for (int i=0;i<size;i++) {
            String ib = toBinary(i,numClasses);
            MultiLabel multiLabel = toML(ib);
            list.add(multiLabel);
        }
        return list;
    }

    private static String toBinary(int number, int length){
        String iBinary = Integer.toBinaryString(number);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<length-iBinary.length();l++){
            sb.append("0");
        }
        sb.append(iBinary);
        String ib = sb.toString();
        return ib;
    }

    private static MultiLabel toML(String str){
        MultiLabel multiLabel = new MultiLabel();
        for (int i=0;i<str.length();i++){
            String sub = str.substring(i,i+1);
            if (sub.equals("1")){
                multiLabel.addLabel(i);
            }
        }
        return multiLabel;
    }
}
