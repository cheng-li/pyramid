package edu.neu.ccs.pyramid.util;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/6/14.
 */
public class DirWalker {
    public static List<File> getFiles(String dir) throws Exception{
        List<File> list = new ArrayList<>();
        addFile(new File(dir), list);
        return list;
    }

    private static void addFile(File file, List<File> list) throws Exception{
        if (file.isDirectory()){
            String[] subNodes = file.list();
//            System.out.println(Arrays.toString(subNodes));
            for (String subNode: subNodes){
                addFile(new File(file,subNode),list);
            }
        } else {
            list.add(file);
        }
    }
}
