package edu.neu.ccs.pyramid.application;

import java.util.Arrays;

/**
 * Created by chengli on 9/2/15.
 */
public class AppLauncher {
    public static void main(String[] args) throws Exception{
        if (args.length==1){
            if (args[0].equals("-help")||(args[0].equals("--help"))){
                help();
            }
            else {
                error();
            }
        } else if (args.length==2){
            launch(args);
        } else {
            error();
        }
    }

    private static void launch(String[] args) throws Exception{
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
                System.err.println("Unknown app name: "+appName);
                System.exit(1);
        }
    }

    private static void help(){
        System.out.println("Usage: ./pyramid <app_name> <properties_file>\n" +
                "The <app_name> is case-insensitive.\n" +
                "The <properties_file> can be specified by either an absolute or a relative path.\n"+
                "Example: ./pyramid app1 config/app1.properties");
        System.exit(0);
    }

    private static void error(){
        System.err.println("Invalid command.\n" +
                "Usage: ./pyramid <app_name> <properties_file>\n" +
                "The <app_name> is case-insensitive.\n" +
                "The <properties_file> can be specified by either an absolute or a relative path.\n"+
                "Example: ./pyramid app1 config/app1.properties");
        System.exit(1);
    }
}
