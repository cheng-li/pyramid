package edu.neu.ccs.pyramid.data_formatter.ohsumed;

import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;

import java.io.File;
import java.util.Scanner;

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


    public static XContentBuilder getBuilder(File file) throws Exception{
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        builder.field("body",getBody(file));
        builder.field("file_name",getFileName(file));
        builder.field("real_label",getExtLabel(file));
        builder.field("label",""+getLabel(file));
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

    static int getLabel(File file) throws Exception{
        String name = file.getParentFile().getName();
        String code = name.substring(1);
        String simplified;
        if (code.startsWith("0")){
            simplified = code.substring(1);
        } else {
            simplified = code;
        }
        //return 0 for C01
        int label = Integer.parseInt(simplified) -1;
        return label;
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

    static String getExtLabel(File file) throws Exception{
        int label = getLabel(file);
        return extLabels[label];
    }


}
