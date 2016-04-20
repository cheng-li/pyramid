package edu.neu.ccs.pyramid.multilabel_classification;

import junit.framework.TestCase;

/**
 * Created by chengli on 4/20/16.
 */
public class EnumeratorTest extends TestCase {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        System.out.println(Enumerator.enumerate(6));
    }

}