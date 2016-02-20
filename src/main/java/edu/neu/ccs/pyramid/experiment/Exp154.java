package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * disk clean up
 * Created by chengli on 2/3/16.
 */
public class Exp154 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);
        String folder = config.getString("folder");
        System.out.println("cleaning "+folder);
        List<File> old = listOld(new File(folder));
        for (File file: old){
            if (file.isDirectory()){
                throw new RuntimeException("is dir");
            }
            file.delete();
        }
        System.out.println("done");
    }

    public static void clean(File folder){
        System.out.println("cleaning "+folder);
        List<File> old = listOld(folder);
        for (File file: old){
            if (file.isDirectory()){
                throw new RuntimeException("is dir");
            }
            file.delete();
        }
        System.out.println("done");
    }

    private static List<File> listOld(File folder){
        File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")||name.endsWith(".PIs")||name.endsWith(".gammas")));
        File lastFile = null;
        int lastIter = -1;
        for (File file: modeFiles){
            String[] split = file.getName().split(Pattern.quote("."));
            int iter = Integer.parseInt(split[1]);
            if (iter>lastIter){
                lastIter = iter;
                lastFile = file;
            }
        }
        Set<File> old = new HashSet<>();
        for (File  file: modeFiles){
            old.add(file);
        }
        for (File file: modeFiles){
            String[] split = file.getName().split(Pattern.quote("."));
            int iter = Integer.parseInt(split[1]);
            if (iter==lastIter-1 || iter==lastIter){
                old.remove(file);
                System.out.println("keep "+file);
            }

        }
        List<File> oldList = old.stream().sorted().collect(Collectors.toList());
//        System.out.println("last file = "+lastFile);
//        System.out.println("old files = "+oldList);
        return oldList;
    }
}
