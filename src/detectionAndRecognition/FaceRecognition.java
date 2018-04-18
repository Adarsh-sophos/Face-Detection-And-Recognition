package detectionAndRecognition;

import java.io.File;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

public class FaceRecognition
{
	FaceRecognizer faceRecognizer;
	
	public void trainModel(String modelName)
	{		
		//System.out.println("Current Working Directory = " + System.getProperty("user.dir"));
				
		System.out.println("Preparing data...");
		
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
        System.out.println("Total labels: " + imageLabelsBuffer.capacity() + "\n\n");
        
        if(modelName == "Eigen")
        	faceRecognizer = EigenFaceRecognizer.create();
        
        else if(modelName == "Fisher")
        	faceRecognizer = FisherFaceRecognizer.create();
        
        else if(modelName == "lbph")
        	faceRecognizer = LBPHFaceRecognizer.create();
        
        else
        	faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(allImages, imageLabels);
	}
	
	public int makePrediction(String imageToRecognize)
	{		
        int labelPredicted = HelperFunctions.recognizeFace(imageToRecognize, faceRecognizer);
     
        System.out.println("Predicted label: " + labelPredicted);
        
        return labelPredicted;
	}
	
	public int makePrediction(Mat face)
	{		
        int labelPredicted = HelperFunctions.recognizeFace(face, faceRecognizer);
     
        //System.out.println("Predicted label: " + labelPredicted);
        
        return labelPredicted;
	}
}
