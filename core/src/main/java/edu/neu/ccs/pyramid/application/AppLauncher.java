package edu.neu.ccs.pyramid.application;

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import edu.neu.ccs.pyramid.Version;
import edu.neu.ccs.pyramid.configuration.Config;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Launch an app based on the string name of the app
 * Created by chengli on 9/2/15.
 */

public class AppLauncher {

    public static void main(String[] args) throws Exception{
        if (args.length==1){
            launch(args);
        } else {
            error();
        }
    }

    private static void launch(String[] args) throws Exception{
        Config config = new Config(args[0]);
        String className = config.getString("pyramid.class");
        String[] mainArgs = Arrays.copyOfRange(args,0,1);
        String realName = matchClass(className);
        if (realName==null){
            System.err.println("Unknown app name: "+className);
            System.exit(1);
        }
        invokeMain(realName,mainArgs);
    }


    private static void error(){
        System.err.println("Invalid command.\n" +
                "Usage: ./pyramid <properties_file>\n" +
                "The <properties_file> can be specified by either an absolute or a relative path.\n"+
                "Example: ./pyramid config/welcome.properties");
        System.exit(1);
    }



    private static String matchClass(String className) throws Exception{
        // deal with normal names
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



}
