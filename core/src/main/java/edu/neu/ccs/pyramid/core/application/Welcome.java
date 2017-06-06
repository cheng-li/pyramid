package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;

/**
 * Created by chengli on 9/5/15.
 */
public class Welcome {
    public static void main(String[] args) {
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        String name = config.getString("yourName");
        System.out.println("Hello "+name+". Welcome to the world of pyramid!");
    }
}
