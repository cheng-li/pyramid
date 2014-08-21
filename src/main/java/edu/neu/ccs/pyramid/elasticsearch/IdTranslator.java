package edu.neu.ccs.pyramid.elasticsearch;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by chengli on 8/20/14.
 */
public class IdTranslator implements Serializable {
    private static final long serialVersionUID = 1L;
    private Map<Integer,String> algorithmToIndex;
    private Map<String, Integer> indexToAlgorithm;

    public IdTranslator() {
        algorithmToIndex = new ConcurrentHashMap<>();
        indexToAlgorithm = new ConcurrentHashMap<>();
    }

    public int numData(){
        return algorithmToIndex.size();
    }

    public String[] dataIndexIds(){

        return indexToAlgorithm.keySet().toArray(new String[0]);

    }

    public void addData(String dataIndexId, int dataAlgorithmId){
        algorithmToIndex.put(dataAlgorithmId,dataIndexId);
        indexToAlgorithm.put(dataIndexId,dataAlgorithmId);
    }

    public int toAlgorithmId(String dataIndexId){
        return indexToAlgorithm.get(dataIndexId);
    }

    public String toIndexId(int dataAlgorithmId){
        return algorithmToIndex.get(dataAlgorithmId);
    }

    public void serialize(File file) throws Exception{
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

    public void serialize(String file) throws Exception{
        File file1 = new File(file);
        serialize(file1);
    }

    public static IdTranslator deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            IdTranslator idTranslator = (IdTranslator)objectInputStream.readObject();
            return idTranslator;
        }
    }

    public static IdTranslator deserialize(String file) throws Exception{
        File file1 = new File(file);
        return deserialize(file1);
    }
}
