�������Ƕȼ��

������
1. libmtcnn.so
��������

2. ģ���ļ�
det1.bin
det1.param
det2.bin
det2.param
det3.bin
det3.param

ʹ�÷���
Mtcnn.java    �����ͽǶȼ��
Box.java    ������Ϣ, box: ������ score: ������������ landmark: �ؼ������� angles: �����Ƕ�(��б��,ˮƽת��,������)
Mtcnn.setMinSizeΪ������С���������С
ʹ�������Ҫ����Mtcnn.releaseDetect�ͷ���Դ

Mtcnn mtcnn = new Mtcnn("model_path", 90); //90Ϊ��С���������С
String img_name = "test.jpg";
BufferedImage img_buff = ImageIO.read(new File(img_name));
Vector<Box> boxes = mtcnn.detectFaces(img_buff); 
mtcnn.releaseDetect();