package edu.neu.ccs.pyramid.tmp;

/**
 * Created by chengli on 9/15/14.
 */

public class Tmp4 {
    public static void main(String[] args) {
        String old = "old";
        String abc = new String(old);
        String another = "haha";

        abc.concat(" adfs ");
        System.out.println(old);
        System.out.println(abc);
        abc = another;
        System.out.println(abc);
    }
}
