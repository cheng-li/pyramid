package edu.neu.ccs.pyramid.classification.ecoc;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ClassifierFactory;
import edu.neu.ccs.pyramid.classification.TrainConfig;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/4/14.
 */
public class ECOC implements Classifier{
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private CodeMatrix codeMatrix;
    private transient ClfDataSet binaryDataSet;
    private String archive;
    private List<Classifier> classifiers;
    private transient int[] originalLabels;
    private transient int numDataPoints;
    private ClassifierFactory classifierFactory;
    private transient TrainConfig trainConfig;

    public ECOC(ECOCConfig ecocConfig,
                ClfDataSet dataSet,
                String archive,
                ClassifierFactory classifierFactory,
                TrainConfig trainConfig) {
        this.binaryDataSet = DataSetUtil.changeLabels(dataSet,2);
        this.numDataPoints = dataSet.getNumDataPoints();
        this.classifierFactory = classifierFactory;
        this.trainConfig = trainConfig;
        this.archive = archive;
        this.numClasses = dataSet.getNumClasses();
        if (ecocConfig.getCodeType()== CodeMatrix.CodeType.EXHAUSTIVE) {
            this.codeMatrix = CodeMatrix.exhaustiveCodes(numClasses);
        } else if (ecocConfig.getCodeType()== CodeMatrix.CodeType.RANDOM){
            this.codeMatrix = CodeMatrix.randomCodes(numClasses,ecocConfig.getNumFunctions());
        } else {
            throw new IllegalArgumentException("unknown code type");
        }
        this.originalLabels = new int[this.numDataPoints];
        System.arraycopy(dataSet.getLabels(),0,this.originalLabels,0,this.numDataPoints);
    }

    public CodeMatrix getCodeMatrix() {
        return codeMatrix;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    public void train() throws Exception{
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        int numFunctions = this.codeMatrix.getNumFunctions();
        System.out.println("total number of ECOC functions = "+numFunctions);
        for (int i=0;i<numFunctions;i++){
            System.out.println("training classifier "+i);
            Classifier classifier = this.trainOneClassifier(i);
            classifier.serialize(new File(this.archive,"model_"+i));
            System.out.println("classifier "+i+" done.");
            System.out.println(stopWatch);
        }
    }

    public int predict(Vector vector){
        int numFunctions = this.classifiers.size();
        int[] code = new int[numFunctions];
        for (int i=0;i<numFunctions;i++){
            Classifier classifier = this.classifiers.get(i);
            int pred = classifier.predict(vector);
            code[i] = pred;
        }
        return this.codeMatrix.matchClass(code);
    }



    public static ECOC deserialize(File savedModel) throws Exception{
        ECOC ecoc;
        try(
                FileInputStream fileInputStream = new FileInputStream(savedModel);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            ecoc = (ECOC)objectInputStream.readObject();
        }
        //read classifiers
        int numFunctions = ecoc.codeMatrix.getNumFunctions();
        ecoc.classifiers = new ArrayList<>(numFunctions);
        for (int i=0;i<numFunctions;i++){
            Classifier classifier = ecoc.classifierFactory.deserialize(new File(ecoc.archive,
                    "model_"+i));
            ecoc.classifiers.add(classifier);
        }
        return ecoc;
    }

    private Classifier trainOneClassifier(int functionIndex)throws Exception{
        int[] aggregatedLabels = this.codeMatrix.aggregateLabels(functionIndex,
                this.originalLabels);
        for (int i=0;i<this.numDataPoints;i++){
            binaryDataSet.setLabel(i, aggregatedLabels[i]);
        }
        return this.classifierFactory.train(binaryDataSet,this.trainConfig);
    }




}
