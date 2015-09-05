package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * loop on Exp97
 * Created by chengli on 5/5/15.
 */
public class Exp98 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        List<Integer> list = config.getIntegers("indices");

        System.out.println(list);

        Config exp97Config = new Config();

        for (int i:list){
            System.out.println(""+i);
            exp97Config.setString("input.folder",new File(config.getString("input.folder"),""+i).getAbsolutePath());
            exp97Config.setString("output.folder",new File(config.getString("output.folder"),""+i).getAbsolutePath());
            Exp97.mainFromConfig(exp97Config);
        }

    }
}
