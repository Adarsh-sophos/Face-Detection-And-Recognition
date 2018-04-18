package detectionAndRecognition;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.opencv_core;

import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import utils.MatrixAndImageConverter;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class FaceDetectionAndRecognition
{
	@FXML
	CheckBox LBPH;
	@FXML
	CheckBox EIGEN;
	@FXML
	CheckBox FISHER;
	
	@FXML
	Button cameraButton;
	@FXML
	ImageView originalFrame;
	@FXML
	CheckBox haarClassifier;
	
	@FXML
	Button trainModel;
	
	@FXML
	Label modelTrained;
	
	@FXML
	CheckBox lbpClassifier;
	
	ScheduledExecutorService timer;
	
	boolean fisherModelTrained = false;
	boolean eigenModelTrained = false;
	boolean lbphModelTrained = false;
	
	VideoCapture videoStreamCapture;
	boolean cameraActive;
	
	CascadeClassifier faceCascade;
	
	FaceRecognition fisherFaceRecognition = new FaceRecognition();
	FaceRecognition eigenFaceRecognition = new FaceRecognition();
	FaceRecognition lbphFaceRecognition = new FaceRecognition();
	
	int faceSize;
	
	void init()
	{
		videoStreamCapture = new VideoCapture();
		faceCascade = new CascadeClassifier();
		faceSize = 0;
		
		originalFrame.setFitWidth(600);
		
		originalFrame.setPreserveRatio(true);
		
		testAccuracy();
		//faceRecognition.trainModel();
	}
	
	void testAccuracy()
	{
		testModelAccuracy fisherRecognition = new testModelAccuracy();
		testModelAccuracy eigenRecognition = new testModelAccuracy();
		testModelAccuracy lbphRecognition = new testModelAccuracy();
		
		fisherRecognition.trainModel("Fisher");
		eigenRecognition.trainModel("Eigen");
		lbphRecognition.trainModel("lbph");
		
		fisherRecognition.accuracy("Fisher");
		eigenRecognition.accuracy("Eigen");
		lbphRecognition.accuracy("lbph");
	}
	
	
	
	Mat grabFrame()
	{
		Mat frame = new Mat();
		
		if (videoStreamCapture.isOpened())
		{
			try
			{
				videoStreamCapture.read(frame);
				
				if (!frame.empty())
				{
					Core.flip(frame, frame, 1);
					detectAndDisplay(frame);
				}
			}
			
			catch (Exception e)
			{
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		if(faceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				faceSize = Math.round(height * 0.2f);
			}
		}
		
		faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(faceSize, faceSize), new Size());
				
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
		{
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
	
			Mat faceExtracted = new Mat(grayFrame, facesArray[i]);
			
			Size sz = new Size(125, 150);
			Mat face = new Mat();
			Imgproc.resize(faceExtracted, face, sz);
			
			//Imgproc.cvtColor(faceExtracted, face, Imgproc.COLOR_BGR2GRAY);
	        
	        BufferedImage image = matToBufferedImage(face);
	        opencv_core.Mat faceJavaCV = bufferedImageToMat(image);
	        
	        Map <Integer, String> map = new HashMap<Integer, String>();
			map.put(1, "saksham");
			map.put(2, "adarsh");
			map.put(3, "mohit");
			map.put(4, "sameer");
					
	        int predictedLabel;
	        
	        if(EIGEN.isSelected())
	        	predictedLabel = eigenFaceRecognition.makePrediction(faceJavaCV);
	        else if(FISHER.isSelected())
	        	predictedLabel = fisherFaceRecognition.makePrediction(faceJavaCV);
	        else if(LBPH.isSelected())
	        	predictedLabel = lbphFaceRecognition.makePrediction(faceJavaCV);
	        else
	        	predictedLabel = -1;
	        
	        String box_text = "Prediction = " + map.get(predictedLabel);
	        
	        int pos_x = (int)Math.max(facesArray[i].tl().x - 10, 0);
	        int pos_y = (int)Math.max(facesArray[i].tl().y - 10, 0);
	        
	        Imgproc.putText(frame, box_text, new Point(pos_x, pos_y), Core.FONT_HERSHEY_PLAIN, 1.5, new Scalar(0, 0, 255, 4.0));
		}
	}
	
	public BufferedImage matToBufferedImage(Mat frame)
	{       
        int type = 0;
        if (frame.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        } else if (frame.channels() == 3) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        BufferedImage image = new BufferedImage(frame.width() ,frame.height(), type);
        WritableRaster raster = image.getRaster();
        DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        byte[] data = dataBuffer.getData();
        frame.get(0, 0, data);
        return image;
    }
	
	public opencv_core.Mat bufferedImageToMat(BufferedImage bi)
	{
        OpenCVFrameConverter.ToMat cv = new OpenCVFrameConverter.ToMat();
        return cv.convertToMat(new Java2DFrameConverter().convert(bi)); 
    }
	
	@FXML
	void haarSelected(Event event)
	{
		if(!lbpClassifier.isSelected() && !haarClassifier.isSelected())
		{
			EIGEN.setDisable(true);
			LBPH.setDisable(true);
			FISHER.setDisable(true);
			//cameraButton.setDisable(true);
			return;
		}
		
		if(lbpClassifier.isSelected())
			lbpClassifier.setSelected(false);
			
		checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
	}
	
	@FXML
	void lbpSelected(Event event)
	{
		if(!lbpClassifier.isSelected() && !haarClassifier.isSelected())
		{
			EIGEN.setDisable(true);
			LBPH.setDisable(true);
			FISHER.setDisable(true);
			//cameraButton.setDisable(true);
			return;
		}
		
		if(haarClassifier.isSelected())
			haarClassifier.setSelected(false);
			
		checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
	}
	
	@FXML 
	void fisherSelected(Event event)
	{
		if(!FISHER.isSelected())
		{
			cameraButton.setDisable(true);
			return;
		}
		
		if(EIGEN.isSelected())
			EIGEN.setSelected(false);
		
		if(LBPH.isSelected())
			LBPH.setSelected(false);
		
		recognizerSelected();
	}
	
	@FXML
	void eigenSelected(Event event)
	{
		if(!EIGEN.isSelected())
		{
			cameraButton.setDisable(true);
			return;
		}
		
		if(FISHER.isSelected())
			FISHER.setSelected(false);
		
		if(LBPH.isSelected())
			LBPH.setSelected(false);
		
		recognizerSelected();
	}
	
	@FXML
	void lbphSelected(Event event)
	{
		if(!LBPH.isSelected())
		{
			cameraButton.setDisable(true);
			return;
		}
		
		if(EIGEN.isSelected())
			EIGEN.setSelected(false);
		
		if(FISHER.isSelected())
			FISHER.setSelected(false);
		
		recognizerSelected();
	}
	
	void checkboxSelection(String classifierPath)
	{
		faceCascade.load(classifierPath);
		
		EIGEN.setDisable(false);
		LBPH.setDisable(false);
		FISHER.setDisable(false);
	}
	
	void recognizerSelected()
	{
		cameraButton.setDisable(false);
	}
	
	@FXML
	void trainTheModel()
	{
		if(EIGEN.isSelected())
		{
			if(eigenModelTrained == false)
			{
				trainModel.setDisable(true);
				System.out.println("Training Eigen Face Recognizer Model...");
				eigenFaceRecognition.trainModel("eigen");
				trainModel.setDisable(false);
				
				eigenModelTrained = true;
				modelTrained.setText("Eigen Model Trained");
			}
			else
				modelTrained.setText("Eigen model is already Trained");
		}
		
		else if(FISHER.isSelected())
		{
			if(fisherModelTrained == false)
			{
				trainModel.setDisable(true);
				System.out.println("Training Fisher Face Recognizer Model...");
				fisherFaceRecognition.trainModel("fisher");
				trainModel.setDisable(false);
				
				fisherModelTrained = true;
				modelTrained.setText("Fisher Model Trained");
			}
			else
				modelTrained.setText("Fisher model is already Trained");
		}
		
		else if(LBPH.isSelected())
		{
			if(lbphModelTrained == false)
			{
				trainModel.setDisable(true);
				System.out.println("Training LBPH Face Recognizer Model...");
				lbphFaceRecognition.trainModel("lbph");
				trainModel.setDisable(false);
				
				lbphModelTrained = true;
				modelTrained.setText("LBPH Model Trained");
			}
			else
				modelTrained.setText("LBPH model is already Trained");
		}
	}
	
	@FXML
	void startCamera()
	{	
		if (!cameraActive)
		{
			haarClassifier.setDisable(true);
			lbpClassifier.setDisable(true);
			EIGEN.setDisable(true);
			FISHER.setDisable(true);
			LBPH.setDisable(true);
			trainModel.setDisable(true);
			
			videoStreamCapture.open(0);
			
			if (videoStreamCapture.isOpened())
			{
				cameraActive = true;
				
				Runnable frameGrabber = new Runnable()
				{					
					@Override
					public void run()
					{
						Mat frame = grabFrame();
						Image imageToShow = MatrixAndImageConverter.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				timer = Executors.newSingleThreadScheduledExecutor();
				timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				cameraButton.setText("Stop Camera");
			}
			
			else
				System.err.println("Failed...");
		}
		
		else
		{
			haarClassifier.setDisable(false);
			lbpClassifier.setDisable(false);
			EIGEN.setDisable(false);
			FISHER.setDisable(false);
			LBPH.setDisable(false);
			trainModel.setDisable(false);
			cameraActive = false;
			cameraButton.setText("Start Camera");
			
			stopAcquisition();
		}
	}
	
	void stopAcquisition()
	{
		if (timer!=null && !timer.isShutdown())
		{
			try
			{
				timer.shutdown();
				timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				System.err.println("Exception in stopping the frame videoStreamCapture, trying to release the camera now... " + e);
			}
		}
		
		if (videoStreamCapture.isOpened())
			videoStreamCapture.release();
	}
	
	private void updateImageView(ImageView view, Image image)
	{
		MatrixAndImageConverter.onFXThread(view.imageProperty(), image);
	}
	
	protected void setClosed()
	{
		stopAcquisition();
	}
}






