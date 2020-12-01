package com.example.facemaskdetector;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/**
 * Created by rohithkvsp on 4/22/18.
 */

public class Classifier {

    private static final String TAG = "TfLite";
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_CHANNEL_SIZE =3;

    private static final int  DIM_HEIGHT =224;
    private static final int DIM_WIDTH = 224;
    private static final int BYTES =4;

     Interpreter tflite;

    private static int digit = -1;
    private static float  prob = 0.0f;

    protected ByteBuffer imgData = null;
    private float[][] ProbArray = null;
    protected String ModelFile = "mobilefacenet_99.61_quantized.tflite";

    //allocate buffer and create interface
    Classifier(Activity activity) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.d(TAG, "Classifier: Adding GPU Delegate");
        } else {
            // if the GPU is not supported, run on 8 threads
            options.setNumThreads(8);
            Log.d(TAG, "Classifier: Not using GPU");
        }
        tflite = new Interpreter(loadModelFile(activity), options);
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_CHANNEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        ProbArray = new float[1][2];
        Log.d(TAG, " Tensorflow Lite Classifier.");
    }

    //load model
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(ModelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    //classify mat
    public int classifyMat(Mat mat) {

        int prediction=0;
        long startTime = SystemClock.uptimeMillis();
        if(tflite!=null) {

            convertMattoTfLiteInput(mat);
            prediction = runInference();
        }

        long endTime = SystemClock.uptimeMillis();
        return prediction;
    }
    //convert opencv mat to tensorflowlite input
    private void convertMattoTfLiteInput(Mat mat)
    {
        imgData.rewind();
        int pixel = 0;
        for (int i = 0; i < DIM_HEIGHT; i++) {
            for (int j = 0; j < DIM_WIDTH; j++) {
                for (int k =0; k < DIM_CHANNEL_SIZE ; k++)
                {
                    imgData.putFloat((float)mat.get(i,j)[k]);
                }

            }
        }
    }

    //run interface
    private int runInference() {
        if(imgData != null)
            tflite.run(imgData, ProbArray);
        return maxProbIndex(ProbArray[0]);
    }
    // find max prob and digit
    private  int maxProbIndex(float[] probs) {
        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        prob = maxProb;
        digit = maxIndex;
        return maxIndex;
    }
    //get predicted digit
    public int getdigit()
    {

        return digit;
    }
    //get predicted  prob
    public float getProb()
    {

        return prob;
    }
    //close interface
    public void close() {
        if(tflite!=null)
        {
            tflite.close();
            tflite = null;
        }

    }
}