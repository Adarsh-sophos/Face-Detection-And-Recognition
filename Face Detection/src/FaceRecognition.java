import java.io.File;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;

public class FaceRecognition
{
	FaceRecognizer faceRecognizer;
	
	public FaceRecognizer trainModel()
	{
		//System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//System.out.println("Working Directory = " + System.getProperty("user.dir"));
		
		HelperFunctions h = new HelperFunctions();
		
		System.out.println("Preparing data...");
		
		String trainingPath = "trainingImages";
		
		int numberOfTrainingImages = h.countTrainingImages(trainingPath);
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
                		
                		h.detectFace(imageMatrix);
                		
                		countImages++;
                		
                		//System.out.println(imagePath);
                	}
                }
            }
        }
        
        System.out.println("Data prepared");
        
        System.out.println("Total faces: " + allImages.size());
        System.out.println("Total labels: " + imageLabelsBuffer.capacity());
        
        // faceRecognizer = FisherFaceRecognizer.create();
        // faceRecognizer = EigenFaceRecognizer.create();
        faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(allImages, imageLabels);
	}
	
	public int makePrediction(String imageToRecognize)
	{
        int labelPredicted = h.recognizeFace("aree3.png", faceRecognizer);
     
        System.out.println("Predicted label: " + labelPredicted);
        
        return labelPredicted;
	}
}
