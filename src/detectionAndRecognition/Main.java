package detectionAndRecognition;

import org.opencv.core.Core;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.fxml.FXMLLoader;

public class Main extends Application
{
	@Override
	public void start(Stage primaryStage)
	{
		try
		{
			FXMLLoader loader = new FXMLLoader(getClass().getResource("FaceDetection.fxml"));
			Pane root = (Pane) loader.load();
			root.setStyle("-fx-background-color: whitesmoke;");
			Scene scene = new Scene(root, 800, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			
			primaryStage.setTitle("Face Detection and Tracking");
			primaryStage.setScene(scene);
			primaryStage.show();
			
			FaceDetectionAndRecognition controller = loader.getController();
			controller.init();
			
			primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>()
			{
				public void handle(WindowEvent we)
				{
					controller.setClosed();
				}
			}));
		}
		
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		launch(args);
	}
}



