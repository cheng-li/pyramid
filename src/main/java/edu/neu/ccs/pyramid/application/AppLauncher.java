package edu.neu.ccs.pyramid.application;

import java.util.Arrays;

/**
 * Created by chengli on 9/2/15.
 */
public class AppLauncher {
    public static void main(String[] args) throws Exception{
        if (args.length!=2){
            System.err.println("Incorrect command format. Usage: ./pyramid <app_name> <config_file>");
            System.exit(1);
        }


        String appName = args[0];
        String appNameLower = appName.toLowerCase();
        String[] appArgs = Arrays.copyOfRange(args,1,2);

        switch (appNameLower){
            case "app1":
                App1.main(appArgs);
                break;
            case "app2":
                App2.main(appArgs);
                break;
            case "app3":
                App3.main(appArgs);
                break;
            default:
                System.err.println("Unknown app name "+appName+" .");
                System.exit(1);
        }

    }
}
