package edu.neu.ccs.pyramid.data_formatter.twenty_newsgroup;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by chengli on 11/14/14.
 */
public class IndexBuilder {

    static String[] extLabels={"alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
            "misc.forsale",  "rec.autos", "rec.motorcycles", "rec.sport.baseball",
            "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med", "sci.space",
            "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
            "talk.politics.misc", "talk.religion.misc"};
    static Map<String,Integer> map = createMap();



    public static XContentBuilder getBuilder(File file, int id) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.field("real_label",getExtLabel(file));
        builder.field("label",""+getLabel(file));
        builder.field("split", getTrainOrTestFixed(file));
        builder.endObject();
        return builder;
    }



    static String getBody(File file) throws Exception{
        String entireFileText = new Scanner(file)
                .useDelimiter("\\A").next();
        return entireFileText;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    static int getLabel(File file) throws Exception{
        String name = file.getParentFile().getName();
        return map.get(name);
    }

    /**
     * random partition
     * @param id
     * @return
     * @throws Exception
     */
    static String getSplit(int id) throws Exception{
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

    /**
     * fixed
     * @param file
     * @return
     * @throws Exception
     */
    static String getTrainOrTestFixed(File file) throws Exception{
        String name = file.getParentFile().getParentFile().getName();
        String res = null;
        if (name.endsWith("train")){
            res = "train";
        } else {
            res = "test";
        }
        return res;
    }

    static String getExtLabel(File file) throws Exception{
        return file.getParentFile().getName();
    }



    static Map<String,Integer> createMap(){
        Map<String,Integer> map = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            map.put(extLabels[i],i);
        }
        return map;
    }
}
