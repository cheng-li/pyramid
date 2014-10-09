package edu.neu.ccs.pyramid.data_formatter.ohsumed;

import edu.neu.ccs.pyramid.util.DirWalker;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.*;

/**
 * Created by chengli on 10/1/14.
 */
public class IndexBuilder {
    static String[] extLabels={
            "Bacterial Infections and Mycoses",
            "Virus Diseases",
            "Parasitic Diseases",
            "Neoplasms",
            "Musculoskeletal Diseases",
            "Digestive System Diseases",
            "Stomatognathic Diseases",
            "Respiratory Tract Diseases",
            "Otorhinolaryngologic Diseases",
            "Nervous System Diseases",
            "Eye Diseases",
            "Urologic and Male Genital Diseases",
            "Female Genital Diseases and Pregnancy Complications",
            "Cardiovascular Diseases",
            "Hemic and Lymphatic Diseases",
            "Neonatal Diseases and Abnormalities",
            "Skin and Connective Tissue Diseases",
            "Nutritional and Metabolic Diseases",
            "Endocrine Diseases",
            "Immunologic Diseases",
            "Disorders of Environmental Origin",
            "Animal Diseases",
            "Pathological Conditions, Signs and Symptoms"
    };


    public static XContentBuilder getBuilder(File file, Map<String, Set<String>> nameToCodesMap) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.array("codes",getCodes(file,nameToCodesMap));
        builder.array("real_labels", getExtLabels(file, nameToCodesMap));
        builder.array("multi_label",getLabels(file,nameToCodesMap));
        builder.field("split",getTrainOrTest(file));
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

    static String[] getLabels(File file, Map<String, Set<String>> nameToCodesMap) throws Exception{
        String name = file.getName();
        return nameToCodesMap.get(name)
                .stream().sorted().map(IndexBuilder::codeToLabel).map(label -> ""+label)
                .toArray(String[]::new);
    }

    static String[] getCodes(File file, Map<String, Set<String>> nameToCodesMap) throws Exception{
        String name = file.getName();
        return nameToCodesMap.get(name)
                .stream().sorted()
                .toArray(String[]::new);
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

    static String[] getExtLabels(File file,Map<String, Set<String>> nameToCodesMap) throws Exception{
        String name = file.getName();
        return nameToCodesMap.get(name)
                .stream().sorted().map(IndexBuilder::codeToLabel).map(label -> extLabels[label])
                .toArray(String[]::new);
    }

    public static int codeToLabel(String code){
        String withoutC = code.substring(1);
        String simplified;
        if (withoutC.startsWith("0")){
            simplified = withoutC.substring(1);
        } else {
            simplified = withoutC;
        }
        //return 0 for C01
        int label = Integer.parseInt(simplified) -1;
        return label;
    }

    public static Map<String, Set<String>> collectCodes(String folder) throws Exception{
        Map<String, Set<String>> map = new HashMap<>();
        List<File> files = DirWalker.getFiles(folder);
        for (File file: files){
            String name = file.getName();
            String labelCode = file.getParentFile().getName();
            if (!map.containsKey(name)){
                map.put(name, new HashSet<>());
            }
            map.get(name).add(labelCode);
        }
        return map;
    }


}
