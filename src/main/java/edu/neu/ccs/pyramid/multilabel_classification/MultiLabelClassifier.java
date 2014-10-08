package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public interface MultiLabelClassifier extends Serializable{
    MultiLabel predict(FeatureRow featureRow);
    default List<MultiLabel> predict(MultiLabelClfDataSet dataSet){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> predict(dataSet.getFeatureRow(i)))
                .collect(Collectors.toList());
    }

    default void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    default void serialize(String file) throws Exception{
        serialize(new File(file));
    }

}
