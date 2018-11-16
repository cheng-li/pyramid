package edu.neu.ccs.pyramid.util;

import java.io.*;
import java.nio.file.Path;

/**
 * Created by chengli on 4/4/15.
 */
public class Serialization {

    public static Object deepCopy(Object serializableObj) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(bos);
        out.writeObject(serializableObj);

        //De-serialization of object
        ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
        ObjectInputStream in = new ObjectInputStream(bis);
        Object copied = in.readObject();
        return copied;
    }

    public static void serialize(Object object, File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(object);
        }
    }

    public static void serialize(Object object, String file) throws Exception{
        serialize(object,new File(file));
    }

    public static void serialize(Object object, Path path) throws Exception{
        serialize(object,path.toFile());
    }


    public static Object deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            return objectInputStream.readObject();
        }
    }

    public static Object deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }

    public static Object deserialize(Path path) throws Exception{
        return deserialize(path.toFile());
    }


}
