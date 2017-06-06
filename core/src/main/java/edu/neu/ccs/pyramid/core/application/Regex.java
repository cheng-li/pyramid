package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;

/**
 * This code is based on the Oracle Java tutorial of regular expression
 * https://docs.oracle.com/javase/tutorial/essential/regex/test_harness.html
 * Created by chengli on 9/10/16.
 */
public class Regex {

    public static void main(String[] args){
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);


        boolean match = config.getString("string").matches(config.getString("regularExpression"));
        if (match){
            System.out.println("match!");
        } else {
            System.out.println("not match!");
        }

    }


}
