package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

/**
 * given dataset level folder, clean up subfolders
 * Created by chengli on 2/3/16.
 */
public class Exp155 {
    public static void main(String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        System.out.println(config);
        File folder = new File(config.getString("folder"));
        File[] subs = folder.listFiles();
        for (File sub: subs){
            File modelFolder = new File(sub,"model");
            Exp154.clean(modelFolder);
        }
    }
}
