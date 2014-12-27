package edu.neu.ccs.pyramid.data_formatter.wipo;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by chengli on 12/27/14.
 */
public class IndexBuilder {
    private String section;
    private String[] extLabels;
    private Map<String,Integer> map;
    private File trainFile;
    private File testFile;

    public IndexBuilder(File trainFile, File testFile, String section) throws Exception{
        this.trainFile = trainFile;
        this.testFile = testFile;
        this.section = section.toUpperCase();
        extLabels = gatherExtLabels();
        map = createMap();
    }

    public boolean accept(String line){
        String mc = line.split("\t")[0].split(",")[0];
        return mc.startsWith(section);
    }


    public String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    public XContentBuilder getBuilder(File file, String line, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(line));
        builder.field("file_name",getFileName(file));
        builder.array("real_label", getRealLabel(line));
        builder.array("label",getLabel(line));
        builder.field("split", getSplit(file,id));
        builder.endObject();
        return builder;
    }

    public String getBody(String line){
        int pos = line.indexOf("\t");
        String res = line.substring(pos+1);
        return res;
    }

    public String getLabel(String line){

        String realLabel = getRealLabel(line);
        int label = map.get(realLabel);
        return ""+label;
    }

    public String getRealLabel(String line){
        return line.split("\t")[0].split(",")[0].substring(0,3);
    }

    public String getSplit(File file, int id) {
        if (file!=trainFile && file!=testFile){
            throw new RuntimeException("file!=trainFile && file!=testFile");
        }

        String res = null;
        if (file == trainFile){
            if ( id%5==0){
                res = "valid";
            } else {
                res = "train";
            }
        } else {
            res = "test";
        }
        return res;
    }

    Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }

    String[] gatherExtLabels() throws Exception{
        Set<String> all = new HashSet<>();
        try(BufferedReader br = new BufferedReader(new FileReader(trainFile))
        ){
            String line;
            while((line = br.readLine())!=null){
                if (accept(line)){
                    all.add(getRealLabel(line));
                }
            }
        }
        return all.toArray(new String[all.size()]);
    }
}
