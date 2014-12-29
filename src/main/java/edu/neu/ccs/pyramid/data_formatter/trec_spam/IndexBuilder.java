package edu.neu.ccs.pyramid.data_formatter.trec_spam;

import org.apache.commons.io.FileUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by chengli on 12/28/14.
 */
public class IndexBuilder {
    private Map<String, String> fileToLabel;

    public IndexBuilder(File labelIndexFile) throws Exception{
        this.fileToLabel = new HashMap<>();
        List<String> lines = FileUtils.readLines(labelIndexFile);
        for (String line: lines){
            String label = line.split(" ")[0];
            String name = line.split(" ")[1].substring(8);
            fileToLabel.put(name,label);
        }
    }

    public String getBody(File file) throws Exception{
        return FileUtils.readFileToString(file);
//        String entireFileText = new Scanner(file)
//                .useDelimiter("\\A").next();
//        return entireFileText;
    }

    public String getFileName(File file) throws Exception{
        String path = file.getParentFile().getName()+"/"+file.getName();
//        System.out.println(path);
        return path;
    }

    public  XContentBuilder getBuilder(File file, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(file));
        builder.array("label",getLabel(file));
        builder.field("split",getTrainOrTest(id));
        builder.endObject();
        return builder;
    }

    private String getLabel(File file) throws Exception{
        String realLabel = getRealLabel(file);
        if (realLabel.equals("ham")){
            return "0";
        } else {
            return "1";
        }
    }

    public  String getRealLabel(File file) throws Exception{
        String fileName = getFileName(file);
        return fileToLabel.get(fileName);
    }

    static String getTrainOrTest(int id) {
        String res = null;
        if ( id%5==0){
            res = "test";
        } else if (id%5==1){
            res = "valid";
        } else {
            res = "train";
        }
        return res;
    }
}
