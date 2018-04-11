package application;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
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

import utils.matToImage;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class FaceDetectionAndRecognition
{
	@FXML
	private Button cameraButton;
	@FXML
	private ImageView originalFrame;
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;
	
	@FXML
	private CheckBox LBPH;
	@FXML
	private CheckBox EIGEN;
	@FXML
	private CheckBox FISHER;
	
	private ScheduledExecutorService timer;
	private VideoCapture capture;
	private boolean cameraActive;
	
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;
	
	FaceRecognition faceRecognition = new FaceRecognition();
	
	protected void init()
	{
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;
		
		// set a fixed width for the frame
		originalFrame.setFitWidth(600);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
		
		faceRecognition.trainModel();
	}
	
	@FXML
	protected void startCamera()
	{	
		if (!this.cameraActive)
		{
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);
			this.EIGEN.setDisable(true);
			this.FISHER.setDisable(true);
			this.LBPH.setDisable(true);
			
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = matToImage.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			this.cameraActive = false;
			this.cameraButton.setText("Start Camera");
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);
			this.EIGEN.setDisable(false);
			this.FISHER.setDisable(false);
			this.LBPH.setDisable(false);
			
			this.stopAcquisition();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// face detection
					this.detectAndDisplay(frame);
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		if (this.absoluteFaceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
				
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
		{
			Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
	
			Mat face = new Mat(grayFrame, facesArray[i]);
	        
	        BufferedImage image = matToBufferedImage(face);
	        opencv_core.Mat faceJavaCV = bufferedImageToMat(image);
	        
	        int predictedLabel = faceRecognition.makePrediction(faceJavaCV);
	        
	        String box_text = "Prediction = " + predictedLabel;
	        
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
	protected void haarSelected(Event event)
	{
		if(!this.lbpClassifier.isSelected() && !this.haarClassifier.isSelected())
		{
			this.EIGEN.setDisable(true);
			this.LBPH.setDisable(true);
			this.FISHER.setDisable(true);
			//this.cameraButton.setDisable(true);
			return;
		}
		
		// check whether the lpb checkbox is selected and deselect it
		if(this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);
			
		this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
	}
	
	@FXML
	protected void lbpSelected(Event event)
	{
		if(!this.lbpClassifier.isSelected() && !this.haarClassifier.isSelected())
		{
			this.EIGEN.setDisable(true);
			this.LBPH.setDisable(true);
			this.FISHER.setDisable(true);
			//this.cameraButton.setDisable(true);
			return;
		}
		
		// check whether the haar checkbox is selected and deselect it
		if(this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);
			
		this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
	}
	
	@FXML 
	private void fisherSelected(Event event)
	{
		if(!this.FISHER.isSelected())
		{
			this.cameraButton.setDisable(true);
			return;
		}
		
		if(this.EIGEN.isSelected())
			this.EIGEN.setSelected(false);
		
		if(this.LBPH.isSelected())
			this.LBPH.setSelected(false);
		
		this.recognizerSelected();
	}
	
	@FXML
	private void eigenSelected(Event event)
	{
		if(!this.EIGEN.isSelected())
		{
			this.cameraButton.setDisable(true);
			return;
		}
		
		if(this.FISHER.isSelected())
			this.FISHER.setSelected(false);
		
		if(this.LBPH.isSelected())
			this.LBPH.setSelected(false);
		
		this.recognizerSelected();
	}
	
	@FXML
	private void lbphSelected(Event event)
	{
		if(!this.LBPH.isSelected())
		{
			this.cameraButton.setDisable(true);
			return;
		}
		
		if(this.EIGEN.isSelected())
			this.EIGEN.setSelected(false);
		
		if(this.FISHER.isSelected())
			this.FISHER.setSelected(false);
		
		this.recognizerSelected();
	}
	
	private void checkboxSelection(String classifierPath)
	{
		// load the classifier(s)
		this.faceCascade.load(classifierPath);
		
		// now the video capture can start
		//this.cameraButton.setDisable(false);
		
		this.EIGEN.setDisable(false);
		this.LBPH.setDisable(false);
		this.FISHER.setDisable(false);
	}
	
	private void recognizerSelected()
	{
		this.cameraButton.setDisable(false);
	}
	
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		matToImage.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}