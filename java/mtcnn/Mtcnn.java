package mtcnn;

import java.awt.image.BufferedImage;
import java.util.Vector;

public class Mtcnn {
    static {
        System.loadLibrary("mtcnn");
    }

    public Mtcnn() {}

    public Mtcnn(String modelPath, int minSize) {
        initDetect(modelPath, minSize);
    }

    private byte[] image2ByteArr(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        byte[] rgb = new byte[w*h*3];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int val = img.getRGB(j, i);
                int red = (val >> 16) & 0xFF;
                int green = (val >> 8) & 0xFF;
                int blue = val & 0xFF;

                rgb[(i*w+j)*3] = (byte) red;
                rgb[(i*w+j)*3+1] = (byte) green;
                rgb[(i*w+j)*3+2] = (byte) blue;
            }
        }

        return rgb;
    }

    public Vector<Box> detectFaces(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        byte[] byteRgb = image2ByteArr(img);
        float[][] result = detect(byteRgb, w, h);

        Vector<Box> boxes = new Vector<Box>();
        for(int i=0; i<result.length; ++i) {
            Box box = new Box();
            box.box[0] = (int)result[i][0];
            box.box[1] = (int)result[i][1];
            box.box[2] = (int)result[i][2];
            box.box[3] = (int)result[i][3];

            for(int j=0; j<10; ++j) {
                box.landmark[j] = result[i][j+4];
            }

            box.score = result[i][14];

            box.calAngles();
            box.limit_square(w, h);
            boxes.add(box);
        }

        return boxes;
    }

    public native boolean initDetect(String model_path, int min_size);
    public native float[][] detect(byte[] image, int width, int height);
    public native void setMinSize(int min_size);
    public native void releaseDetect();
}
