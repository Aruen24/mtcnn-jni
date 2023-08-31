人脸及角度检测

依赖项
1. libmtcnn.so
人脸检测库

2. 模型文件
det1.bin
det1.param
det2.bin
det2.param
det3.bin
det3.param

使用方法
Mtcnn.java    人脸和角度检测
Box.java    人脸信息, box: 人脸框 score: 人脸质量分数 landmark: 关键点坐标 angles: 人脸角度(倾斜角,水平转角,俯仰角)
Mtcnn.setMinSize为设置最小检测人脸大小
使用完后需要调用Mtcnn.releaseDetect释放资源

Mtcnn mtcnn = new Mtcnn("model_path", 90); //90为最小检测人脸大小
String img_name = "test.jpg";
BufferedImage img_buff = ImageIO.read(new File(img_name));
Vector<Box> boxes = mtcnn.detectFaces(img_buff); 
mtcnn.releaseDetect();