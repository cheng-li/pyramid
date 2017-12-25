package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;

public class App3ModelInspector {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        String modelFolder = config.getString("app3ModelFolder");
        App2.CheckPoint checkPoint = (App2.CheckPoint) Serialization.deserialize(new File(modelFolder,"checkpoint"));
        System.out.println("last training iteration = "+checkPoint.getLastIter());

    }
}
