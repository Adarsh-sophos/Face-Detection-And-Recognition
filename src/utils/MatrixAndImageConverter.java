package utils;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import org.opencv.core.Mat;

import javafx.application.Platform;
import javafx.beans.property.ObjectProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;

public final class MatrixAndImageConverter
{	
	public static <T> void onFXThread(final ObjectProperty<T> p, final T v)
	{
		Platform.runLater(() -> {
			p.set(v);
		});
	}
	
	private static BufferedImage matToBufferedImage(Mat original)
	{
		BufferedImage tempImage = null;
		
		int w = original.width();
		int h = original.height();
		int c = original.channels();
		
		byte[] pixelsSource = new byte[w * h * c];
		
		original.get(0, 0, pixelsSource);
		
		if(original.channels() > 1)
			tempImage = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		
		else
			tempImage = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
		
		final byte[] pixelsTarget = ((DataBufferByte) tempImage.getRaster().getDataBuffer()).getData();
		System.arraycopy(pixelsSource, 0, pixelsTarget, 0, pixelsSource.length);
		
		return tempImage;
	}
	
	public static Image mat2Image(Mat frame)
	{
		try
		{
			return SwingFXUtils.toFXImage(matToBufferedImage(frame), null);
		}
		catch (Exception e)
		{
			System.err.println("Cannot convert the Mat obejct: " + e);
			return null;
		}
	}
}







