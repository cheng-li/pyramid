package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.Console;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

        Pattern pattern =
                Pattern.compile(config.getString("regularExpression"));

        Matcher matcher =
                pattern.matcher(config.getString("string"));

        boolean found = false;
        while (matcher.find()) {
            System.out.format("Regular expression matches the text" +
                            " \"%s\" starting at " +
                            "character %d and ending at character %d.%n",
                    matcher.group(),
                    matcher.start()+1,
                    matcher.end());
            found = true;
        }
        if(!found){
            System.out.format("No match found.%n");
        }

    }


}
