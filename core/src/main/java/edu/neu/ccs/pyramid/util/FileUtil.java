package edu.neu.ccs.pyramid.util;

import java.io.File;
import java.nio.file.Paths;

public class FileUtil {

    public static File getFile(String first, String... more){
        File file = Paths.get(first, more).toFile();
        file.getParentFile().mkdirs();
        return file;
    }


    public static File getDir(String first, String... more){
        File file = Paths.get(first, more).toFile();
        file.mkdirs();
        return file;
    }
}
