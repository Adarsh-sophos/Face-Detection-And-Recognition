package detectionAndRecognition;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import java.io.File;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;

public class testModelAccuracy
{
	FaceRecognizer faceRecognizer;
	
	public void trainModel(String modelName)
	{		
		//System.out.println("Current Working Directory = " + System.getProperty("user.dir"));
				
		System.out.println("\nPreparing data...");
		
		String trainingPath = "trainingImages";
		
		int numberOfTrainingImages = HelperFunctions.countTrainingImages(trainingPath);
		MatVector allImages = new MatVector(numberOfTrainingImages);
		
		Mat imageLabels = new Mat(numberOfTrainingImages, 1, CV_32SC1);
        IntBuffer imageLabelsBuffer = imageLabels.createBuffer();
		
		File directory = new File(trainingPath);
        
        File[] listOfDirectories = directory.listFiles();
        
        int countImages = 0;
        
        for(File dirName : listOfDirectories)
        {
            if(dirName.isDirectory())
            {
            	int label = Integer.parseInt(dirName.getName().split("\\-")[1]);
            	
            	//System.out.println(label);
            	
            	String dirPath = trainingPath + "/" + dirName.getName();
            	
                File files = new File(dirPath);
                
                File[] listOfImages = files.listFiles();
                
                for(File image : listOfImages)
                {
                	if(image.isFile())
                	{
                		String imageName = image.getName();
                		
                		if(imageName.startsWith("."))
                			continue;
                		
                		String imagePath = dirPath + "/" + imageName;
                		
                		Mat imageMatrix = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
                		
                		allImages.put(countImages, imageMatrix);
                		
                		imageLabelsBuffer.put(countImages, label);
                		
                		//HelperFunctions.detectFace(imageMatrix);
                		
                		countImages++;
                		
                		//System.out.println(imagePath);
                	}
                }
            }
        }
        
        System.out.println("Data prepared");
        
        System.out.println("Total faces: " + allImages.size());
        System.out.println("Total labels: " + imageLabelsBuffer.capacity());
        
        if(modelName == "Eigen")
        	faceRecognizer = EigenFaceRecognizer.create();
        
        else if(modelName == "Fisher")
        	faceRecognizer = FisherFaceRecognizer.create();
        
        else if(modelName == "lbph")
        	faceRecognizer = LBPHFaceRecognizer.create();
        
        else
        	faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(allImages, imageLabels);
        System.out.println("Model Trained");
	}
	
	public void accuracy(String modelName)
	{
		int correctPredicted = 0;
		int incorrectPredicted = 0;
		
		Map <Integer, String> map = new HashMap<Integer, String>();
		map.put(1, "saksham");
		map.put(2, "adarsh");
		map.put(3, "mohit");
				
		System.out.println("\nTesting data...");
		
		String testPath = "testImages";
				
		File directory = new File(testPath);
        
        File[] listOfImages = directory.listFiles();
        
        for(File image : listOfImages)
        {
        	String personName = image.getName().split("\\-")[0];
        	//System.out.println(personName);
        	
    		if(personName.startsWith("."))
    			continue;
    		
    		String imagePath = testPath + "/" + image.getName();
    		
    		Mat imageMatrix = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
    		
    		int label = HelperFunctions.recognizeFace(imageMatrix, faceRecognizer);
    		
    		if(map.get(label).equals(personName))
    			correctPredicted += 1;
    		else
    			incorrectPredicted += 1;
    		
    		System.out.println("Predicted : " + map.get(label) + "\tActual : " + personName);
        }
        
        //System.out.println(correctPredicted);
        //System.out.println(incorrectPredicted);
        
        float accuracy = (correctPredicted * 100) / (float)(correctPredicted + incorrectPredicted);
        System.out.println(modelName + " accuracy : " + accuracy);
	}
}



