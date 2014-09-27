package edu.neu.ccs.pyramid.dataset;

public class MultiLabelTest {
    public static void main(String[] args) {
        test1();
    }

    static void test1(){
        MultiLabel multiLabel = new MultiLabel(3);
        multiLabel.addLabel(0).addLabel(2);
        System.out.println(multiLabel);
        System.out.println(multiLabel.getLabelForClass(0));
        System.out.println(multiLabel.getLabelForClass(1));
        System.out.println(multiLabel.getLabelForClass(2));
        System.out.println(multiLabel.getMatchedLabels());
    }

    static void test2(){
        MultiLabel multiLabel = new MultiLabel(3);
        multiLabel.addLabel(0).addLabel(3);

    }



}