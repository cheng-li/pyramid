package edu.neu.ccs.pyramid.dataset;


import java.io.File;

public class HierarchicalArffParserTest {

    public static void main(String[] args) throws Exception{
        File file = new File("/Users/chengli/Downloads/reuters/rcv1subset_topics_train_1.arff");
        System.out.println(HierarchicalArffParser.parseNumFeatures(file));
        System.out.println(HierarchicalArffParser.parseNumInstances(file));

        System.out.println(HierarchicalArffParser.loadLabelTranslator(file));

//        MultiLabelClfDataSet dataSet = HierarchicalArffParser.load(file);
//        TRECFormat.save(dataSet,"/Users/chengli/Downloads/reuters/train");

        HierarchicalArffParser.toTrec("/Users/chengli/Downloads/reuters/rcv1subset_topics_train_1.arff","/Users/chengli/Downloads/reuters/train");

        System.out.println(HierarchicalArffParser.getParents("CCAT/C15/C151/C1511"));
        System.out.println(HierarchicalArffParser.getParents("CCAT"));
        System.out.println(HierarchicalArffParser.getParents("CCAT/C15"));
    }

}