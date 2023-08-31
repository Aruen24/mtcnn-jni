package mtcnn;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import javax.imageio.ImageIO;
import java.awt.image.ColorConvertOp;
import java.awt.color.ColorSpace;
import java.util.List;
import java.util.Vector;

public class Test {
    public static byte[] image2ByteArr(BufferedImage img) {
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
                //System.out.println(String.valueOf((i*h+j)*3));
            }
        }
        return rgb;
    }

    public static int[][][] image2FloatArr(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int[][][] floatValues = new int[w][h][3];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int val = img.getRGB(j, i);
                floatValues[j][i][0] = (val >> 16) & 0xFF;
                floatValues[j][i][1] = (val >> 8) & 0xFF;
                floatValues[j][i][2] = val & 0xFF;
                //System.out.println(Arrays.toString(floatValues[j][i]));
            }
        }
        return floatValues;
    }

    public static void main(String args[]) {
        Mtcnn mtcnn = new Mtcnn("/home/chenyong/android_project/tf_face_filter/mtcnn_jni/model", 40);
        //mtcnn.initDetect("/home/chenyong/android_project/tf_face_filter/mtcnn_jni/model", 30, 1);

        String fnm = "/home/chenyong/android_project/tf_face_filter/mtcnn_jni/6.png";
        try {
            List<String> files = new ArrayList<String>();
            File file = new File("../../source/胡天");

            File[] tempList = file.listFiles();
            for (int i = 0; i < tempList.length; i++) {
                if (tempList[i].isFile()) {
                    files.add(tempList[i].toString());
                }
            }

            BufferedImage img_buff=null;
            Vector<Box> boxes=null;
            for(String img_name : files) {
                System.out.println(img_name);
                img_buff = ImageIO.read(new File(img_name));
                boxes = mtcnn.detectFaces(img_buff);

                for (int i = 0; i < boxes.size(); ++i) {
                    System.out.println(Arrays.toString(boxes.get(i).box));
                    System.out.println(Arrays.toString(boxes.get(i).angles));
                }
            }

            //BufferedImage image = ImageIO.read(new File(fnm));

            //BufferedImage rgbImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            //new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_sRGB), null).filter(image, rgbImage);
            //rgbImage.setData(image.getData());
            //byte[] matrixRGB= (byte[]) rgbImage.getData().getDataElements(0, 0, image.getWidth(), image.getHeight(), null);

            //byte[] byteRgb = image2ByteArr(image);

            /*
            for(int i=0; i<byteRgb.length; ++i) {
                if(i%3==0) {
                    System.out.print("\n");
                }
                System.out.print(String.valueOf(byteRgb[i])+" ");
            }*/
            //System.out.println(Arrays.toString(byteRgb));
            //System.out.println(String.valueOf(matrixRGB.length));

            /*
            mtcnn.setMinSize(40);
            Vector<Box> result = mtcnn.detectFaces(image);
            for(int i=0; i<result.size(); ++i) {
                System.out.println(Arrays.toString(result.get(i).box));
                System.out.println(Arrays.toString(result.get(i).landmark));
                System.out.println(Arrays.toString(result.get(i).angles));
            }*/
        } catch (IOException e) {
            e.printStackTrace();
        }
        mtcnn.releaseDetect();
    }
}




