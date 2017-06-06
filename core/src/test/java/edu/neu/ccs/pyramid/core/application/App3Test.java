package edu.neu.ccs.pyramid.core.application;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 11/26/16.
 */
public class App3Test{
    public static void main(String[] args) throws Exception{
        String[] s1 = {"/Users/chengli/tmp/app3-1.properties"};
        String[] s2 = {"/Users/chengli/tmp/app3-2.properties"};
        List<String[]> runs = new ArrayList<>();
        runs.add(s1);
        runs.add(s2);


        runs.stream().parallel().forEach(s -> {
            try {
                App3.main(s);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

}