package edu.neu.ccs.pyramid.dataset;

import java.io.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * a one to one mapping between int ids and ext ids
 * extids can be indexIds
 * duplicate ids are not allowed
 * for duplicate extids, should just save them as extids in dataset, cannot be translated back
 * Created by chengli on 8/20/14.
 */
public class IdTranslator implements Serializable {
    private static final long serialVersionUID = 1L;
    private Map<Integer,String> intToExt;
    private Map<String, Integer> extToInt;

    public IdTranslator() {
        intToExt = new ConcurrentHashMap<>();
        extToInt = new ConcurrentHashMap<>();
    }

    public int numData(){
        return intToExt.size();
    }

    public String[] getAllExtIds(){
        return extToInt.keySet().toArray(new String[0]);
    }

    /**
     * assume extIds are different
     * @param extId
     * @param intId
     */
    public void addData(int intId, String extId){
        if (intToExt.containsKey(intId)){
            throw new IllegalArgumentException(intId+"already exists");
        }
        if (extToInt.containsKey(extId)){
            throw new IllegalArgumentException(extId+"already exists");
        }
        intToExt.put(intId, extId);
        extToInt.put(extId, intId);
    }

    public int toIntId(String extId){
        return extToInt.get(extId);
    }

    public String toExtId(int intId){
        return intToExt.get(intId);
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


    public static IdTranslator newDefaultIdTranslator(int numDataPoints){
        IdTranslator idTranslator = new IdTranslator();
        for (int i=0;i<numDataPoints;i++){
            idTranslator.addData(i,""+i);
        }
        return idTranslator;
    }
}
