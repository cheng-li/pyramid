package edu.neu.ccs.pyramid.dataset;

public class MultiLabelTest {
    public static void main(String[] args)
    {
//        test3();
        test4();
        test5();
        test6();
    }

    static void test1(){
        MultiLabel multiLabel = new MultiLabel(3);
        multiLabel.addLabel(0).addLabel(2);
        System.out.println(multiLabel);
        System.out.println(multiLabel.matchClass(0));
        System.out.println(multiLabel.matchClass(1));
        System.out.println(multiLabel.matchClass(2));
        System.out.println(multiLabel.getMatchedLabels());
    }

    static void test2(){
        MultiLabel multiLabel = new MultiLabel(3);
        multiLabel.addLabel(0).addLabel(3);

    }

    static void test3(){
        MultiLabel multiLabel1 = new MultiLabel(3).addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel(4).addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test4(){
        MultiLabel multiLabel1 = new MultiLabel(3).addLabel(0).addLabel(1);
        MultiLabel multiLabel2 = new MultiLabel(4).addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test5(){
        MultiLabel multiLabel1 = new MultiLabel(3).addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel(3).addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test6(){
        MultiLabel multiLabel1 = new MultiLabel(3).addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel(3).addLabel(0).addLabel(2);
        System.out.println(multiLabel1.equals(multiLabel2));
    }





}