package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 1/4/15.
 */
public class Exp50 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        Config exp49Config = new Config(config.getString("exp49Config"));
        Config exp36Config = new Config(config.getString("exp36Config"));
        Config exp33Config = new Config(config.getString("exp33Config"));
        exp49Config.setString("archive.folder",new File(config.getString("archive.folder"),"exp49").getAbsolutePath());
        exp33Config.setString("archive.folder",new File(config.getString("archive.folder"),"exp33").getAbsolutePath());
        exp36Config.setString("input.folder",exp49Config.getString("archive.folder"));
        exp33Config.setString("input.folder",exp49Config.getString("archive.folder"));




        Set<String> setTypes = new HashSet<>();
        setTypes.add("easy");
        setTypes.add("hard");
        setTypes.add("uncertain");
        setTypes.add("random");
        for (String focusSetType: setTypes){
            exp49Config.setString("extraction.focusSet.type",focusSetType);
            for (String validationSetType: setTypes){
                System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                System.out.println("focusSet="+focusSetType);
                System.out.println("validationSet="+validationSetType);
                exp49Config.setString("extraction.validationSet.type",validationSetType);

                Exp49.mainFromConfig(exp49Config);
                Config bestParams = Exp36.mainFromConfig(exp36Config);
                exp33Config.setDouble("gaussianPriorVariance",bestParams.getDouble("gaussianPriorVariance"));
                System.out.println("focusSet="+focusSetType);
                System.out.println("validationSet="+validationSetType);
                Exp33.mainFromConfig(exp33Config);
            }
        }


    }


}
