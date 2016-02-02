package edu.neu.ccs.pyramid.data_formatter.reuters21578;

import org.apache.commons.io.FileUtils;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.Map;
import java.util.Set;

/**
 * Created by chengli on 2/1/16.
 */
public class IndexBuilder {
    static String getBody(File file) throws Exception{
        String entireFileText = FileUtils.readFileToString(file);
        return entireFileText;
    }

    static String getFileName(File file) throws Exception{
        String fullPath = file.getAbsolutePath();
        return fullPath;
    }

    static String[] getLabels(File file, Map<String, String[]> nameToCodesMap) throws Exception{
        String name = file.getParentFile().getName()+"/"+file.getName();
        return nameToCodesMap.get(name);
    }

    public static XContentBuilder getBuilder(File file, Map<String, String[]> nameToCodesMap) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.array("labels",getLabels(file, nameToCodesMap));
        builder.field("split",getTrainOrTest(file));
        builder.endObject();
        return builder;
    }

    static String getTrainOrTest(File file) throws Exception{
        String name = file.getParentFile().getParentFile().getName();
        String res = null;
        if (name.equals("training")){
            res = "train";
        } else {
            res = "test";
        }
        return res;
    }
}
