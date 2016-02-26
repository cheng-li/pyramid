package edu.neu.ccs.pyramid.application;

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import edu.neu.ccs.pyramid.Version;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Launch an app based on the string name of the app
 * Created by chengli on 9/2/15.
 */

public class AppLauncher {
    private static final String[] helpCommands = {"help","-help","--help",
            "usage","-usage","--usage"};

    private static final String[] versionCommands = {"version","-version","--version"};

    public static void main(String[] args) throws Exception{
        if (args.length==1){
            Set<String> helpSet = Arrays.stream(helpCommands).collect(Collectors.toSet());
            Set<String> versionSet = Arrays.stream(versionCommands).collect(Collectors.toSet());
            String arg = args[0].toLowerCase();
            if (helpSet.contains(arg)){
                help();
            } else if (versionSet.contains(arg)){
                version();
            } else {
                error();
            }
        } else if (args.length==2){
            launch(args);
        } else {
            error();
        }
    }

    private static void launch(String[] args) throws Exception{
        String className = args[0];
        String[] mainArgs = Arrays.copyOfRange(args,1,2);
        String realName = matchClass(className);
        if (realName==null){
            System.err.println("Unknown app name: "+className);
            System.exit(1);
        }
        invokeMain(realName,mainArgs);
    }

    private static void help(){
        System.out.println("Usage: ./pyramid <app_name> <properties_file>\n" +
                "The <app_name> is case-insensitive.\n" +
                "The <properties_file> can be specified by either an absolute or a relative path.\n"+
                "Example: ./pyramid welcome config/welcome.properties");
        System.exit(0);
    }

    private static void error(){
        System.err.println("Invalid command.\n" +
                "Usage: ./pyramid <app_name> <properties_file>\n" +
                "The <app_name> is case-insensitive.\n" +
                "The <properties_file> can be specified by either an absolute or a relative path.\n"+
                "Example: ./pyramid welcome config/welcome.properties");
        System.exit(1);
    }



    private static String matchClass(String className) throws Exception{
        String lower = className.toLowerCase();
        String realName = null;
        ClassPath classPath = ClassPath.from(Thread.currentThread().getContextClassLoader());
        ImmutableSet<ClassPath.ClassInfo> classes = classPath.getTopLevelClasses("edu.neu.ccs.pyramid.application");
        for (ClassPath.ClassInfo classInfo: classes){
            if (classInfo.getSimpleName().toLowerCase().equals(lower)){
                realName = classInfo.getName();
                break;
            }
        }
        return realName;
    }

    private static void invokeMain(String className, String[] args) throws Exception{
        Class<?> c = Class.forName(className);
        Class[] argTypes = new Class[] { String[].class };
        Method main = c.getDeclaredMethod("main", argTypes);
        main.invoke(null, (Object)args);
    }

    private static void version(){
        System.out.println("Pyramid "+ Version.getVersion());
    }


}
