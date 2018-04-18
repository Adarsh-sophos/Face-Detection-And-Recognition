package detectionAndRecognition;

import java.io.File;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.DoublePointer;

import static org.bytedeco.javacpp.opencv_face.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class HelperFunctions
{
	public static int countTrainingImages(String path)
	{
		int numberOfTrainingExamples = 0;
		
		File directory = new File(path);
		File[] listOfDirectories = directory.listFiles();
		
		for(File dirName : listOfDirectories)
		{
            if(dirName.isDirectory())
            {
            	File files = new File(path+"/"+dirName.getName());         
                File[] listOfImages = files.listFiles();
                
                System.out.println(dirName.getName());
                numberOfTrainingExamples += listOfImages.length;
            }
		}
		
		return numberOfTrainingExamples;
	}
	
	public static RectVector detectFace(Mat image)
	{
		Mat imageGray = new Mat();
        
        //cvtColor(image, imageGray, COLOR_BGRA2GRAY);
        
        equalizeHist(imageGray, imageGray);

        CascadeClassifier face_cascade = new CascadeClassifier("resources/haarcascades/haarcascade_frontalface_alt.xml");
        
        RectVector faces = new RectVector();
        
        face_cascade.detectMultiScale(imageGray, faces);
        
        if(faces.size() == 0)
        	return null;
        
        return faces;
	}
	
	public static int recognizeFace(String imageName, FaceRecognizer faceRecognize)
	{
		Mat imageToTest = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        
        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognize.predict(imageToTest, label, confidence);
        
        //System.out.println(confidence.get(0));
        return label.get(0);
	}
	
	public static int recognizeFace(Mat face, FaceRecognizer faceRecognize)
	{
        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognize.predict(face, label, confidence);
        
        //System.out.println(label.get(0) + " " + label.get(1) + " " + label.get(2));
        System.out.println(confidence.get(0));
        return label.get(0);
	}
}
