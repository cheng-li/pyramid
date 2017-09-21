package edu.neu.ccs.pyramid.dataset;

public class MultiLabelTest {
    public static void main(String[] args)
    {
//        test1();
//        test3();
//        test4();
//        test5();
//        test6();
        test9();
    }

    static void test1(){
        MultiLabel multiLabel = new MultiLabel();
        multiLabel.addLabel(0).addLabel(2);
        System.out.println(multiLabel);
        System.out.println(multiLabel.matchClass(0));
        System.out.println(multiLabel.matchClass(1));
        System.out.println(multiLabel.matchClass(2));
        System.out.println(multiLabel.getMatchedLabels());
    }

    static void test2(){
        MultiLabel multiLabel = new MultiLabel();
        multiLabel.addLabel(0).addLabel(3);

    }

    static void test3(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test4(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(1);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test5(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(1);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test6(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(2);
        System.out.println(multiLabel1.equals(multiLabel2));
    }

    static void test7(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2).addLabel(3);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(2).addLabel(4).addLabel(5);
        System.out.println(MultiLabel.symmetricDifference(multiLabel1,multiLabel2));
    }

    static void test8(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2);

        System.out.println(multiLabel1.outOfBound(2));
        System.out.println(multiLabel1.outOfBound(3));
    }

    static void test9(){
        MultiLabel multiLabel1 = new MultiLabel().addLabel(0).addLabel(2).addLabel(1);
        MultiLabel multiLabel2 = new MultiLabel().addLabel(0).addLabel(1).addLabel(2);
        System.out.println(multiLabel1.isSubsetOf(multiLabel2));
        multiLabel1.removeAllLabels();
        System.out.println(multiLabel1);
    }

}